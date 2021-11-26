import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, ConcatDataset
import torchvision.models as models

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from PIL import Image
import urllib
import os
import os.path as osp
import time
import numpy as np

from utils import Params, AverageMeter, accuracy, save_checkpoint, train_epoch, validate, LDAMLoss, get_chunk_idx
import data
from models import SharedEmbedderModel

args = Params()
best_prec3 = 0.0
cudnn.benchmark = True

# model
model = SharedEmbedderModel(num_classes=8142, hidden_size=768, share_embedder=args.share_embedder).cuda()
model.alpha = 1.0
model.inference_alpha = args.inference_alpha

# data ---------------------------------------

if args.share_embedder:
    config = resolve_data_config({}, model=model.embedder)
else:
    config = resolve_data_config({}, model=model.embedder_cb)

train_dataset_full = data.INAT(args.data_root, args.train_file, args.cat_file, config, args.beta, is_train=True,
                               double_img=args.resampling, chunk_size=args.chunk_size)
val_dataset_full = data.INAT(args.data_root, args.val_file, args.cat_file, config, args.beta, is_train=False)

# chunked training set
train_datasets = []
# train_loaders = []
chunks_img_train = get_chunk_idx(train_dataset_full.chunks_classes, train_dataset_full.classes_inv)
for i in range(train_dataset_full.num_chunks):
    train_dataset_i = Subset(train_dataset_full, chunks_img_train[i])
    # train_loader_i = torch.utils.data.DataLoader(train_dataset_i, batch_size=args.batch_size, shuffle=True,
    #                                              num_workers=args.workers, pin_memory=True)
    train_datasets.append(train_dataset_i)
    # train_loaders.append(train_loader_i)

# chunked validation set (same chunking as for training set!)
val_datasets = []
val_loaders = []
chunks_img_val = get_chunk_idx(train_dataset_full.chunks_classes, val_dataset_full.classes_inv)
for i in range(train_dataset_full.num_chunks):
    val_dataset_i = Subset(val_dataset_full, chunks_img_val[i])
    val_loader_i = torch.utils.data.DataLoader(val_dataset_i, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    val_datasets.append(val_dataset_i)
    val_loaders.append(val_loader_i)

# data loader over full validation set
val_loader_full = torch.utils.data.DataLoader(val_dataset_full, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

# data ---------------------------------------

# loss & optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# continue training
if args.resume:
    if os.path.isfile(args.resume):
        print('... loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec3 = checkpoint['best_prec3']
        model.alpha = checkpoint['alpha']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('...... done loading checkpoint {}'.format(args.resume))
    else:
        print('... no checkpoint found at {}'.format(args.resume))

# training loop
# stores the data loaders for chunks with below-threshold validation accuracy & should thus be used for training
train_dataset = train_dataset_full
exit_thresh = args.exit_thresh
for epoch in range(args.start_epoch, args.epochs):

    # set alphas according to schedule
    model.alpha = 1 - (epoch / args.epochs) ** 2
    if args.merged_training:
        model.inference_alpha = model.alpha
    else:
        model.inference_alpha = 0.

    # training epoch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    criterion_reweighted = nn.CrossEntropyLoss(weight=train_dataset_full.class_weights).cuda()
    if args.reweighting:
        train_epoch(args, train_loader, model, criterion, optimizer, epoch, criterion_reweighted)
    else:
        train_epoch(args, train_loader, model, criterion, optimizer, epoch)

    # validation epoch
    prec1, prec3 = validate(args, val_loader_full, model, criterion, False)

    if True:  # TODO: check only every # epochs

        # validate each chunk to decide on which chunks to train in the next epoch
        assert len(val_loaders) == len(train_datasets)
        prec3_per_chunk = []
        for tl_idx, val_loader in enumerate(val_loaders):
            prec1_i, prec3_i = validate(args, val_loader, model, criterion, False)
            prec3_per_chunk.append(prec3_i.item())
        print(prec3_per_chunk)
        current_train_loaders_idx = list(np.where(np.array(prec3_per_chunk) < exit_thresh)[0])
        current_train_datasets = []
        for i in current_train_loaders_idx:
            current_train_datasets.append(train_datasets[i])
        print(len(current_train_datasets))
        train_dataset = ConcatDataset(current_train_datasets)

    # save model
    is_best = prec3 > best_prec3
    best_prec3 = max(prec3, best_prec3)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec3': best_prec3,
        'alpha': model.alpha,
        'optimizer': optimizer.state_dict(),
    }, is_best, args.save_path)

# x = np.arange(0,25)
# y = 1 - (x / 25) ** 2
# plt.plot(x,y)
# plt.xlabel('epoch')
# plt.ylabel(r'$\alpha$')
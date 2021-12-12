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
from math import ceil

from utils import AverageMeter, accuracy, save_checkpoint, train_epoch, validate, LDAMLoss, get_chunk_idx
from params import Params
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
    train_datasets.append(train_dataset_i)

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
down_weight_factors = torch.ones_like(train_dataset_full.class_weights)
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
    if args.weight_both_branches:
        criterion = nn.CrossEntropyLoss(down_weight_factors).cuda()
    criterion_reweighted = nn.CrossEntropyLoss(weight=train_dataset_full.class_weights * down_weight_factors).cuda()
    if args.reweighting:
        train_epoch(args, train_loader, model, criterion, optimizer, epoch, criterion_reweighted)
    else:
        train_epoch(args, train_loader, model, criterion, optimizer, epoch)

    # validation epoch
    prec1, prec3 = validate(args, val_loader_full, model, criterion, False)

    # validate each chunk to decide on which chunks to train in the next epoch
    assert len(val_loaders) == len(train_datasets)
    prec3_per_chunk = []
    for tl_idx, val_loader in enumerate(val_loaders):
        prec1_i, prec3_i = validate(args, val_loader, model, criterion, False)
        prec3_per_chunk.append(prec3_i.item())
    print(prec3_per_chunk)
    if args.exit_strategy == 'downweight_fixed':
        passed_mask = np.array(prec3_per_chunk) > exit_thresh
        current_train_loaders_idx_passed = list(np.where(passed_mask)[0])
        down_weight_factors = torch.ones_like(train_dataset_full.class_weights)
        for i in current_train_loaders_idx_passed:
            down_weight_factors[train_dataset_full.chunks_classes[i]] = args.dw_factor
        print(len(current_train_loaders_idx_passed))
        print(torch.sum(down_weight_factors))
    elif args.exit_strategy == 'downweight_dynamic':
        passed_mask = np.array(prec3_per_chunk) > exit_thresh
        current_train_loaders_idx_passed = list(np.where(passed_mask)[0])
        down_weight_factors = torch.ones_like(train_dataset_full.class_weights)
        for i in current_train_loaders_idx_passed:
            down_weight_factors[train_dataset_full.chunks_classes[i]] = args.dw_factor
        # if all chunks are above the current thresh -> increase it
        if np.all(passed_mask):
            exit_thresh = min(exit_thresh + args.thresh_increase, args.max_thresh)
        print('exit_thresh: {}'.format(exit_thresh))
        print(len(current_train_loaders_idx_passed))
        print(torch.sum(down_weight_factors))
    elif args.exit_strategy == 'downweight_min':
        exit_thresh = min(ceil(np.min(np.array(prec3_per_chunk)) / args.nearest) * args.nearest, args.max_thresh)
        passed_mask = np.array(prec3_per_chunk) > exit_thresh
        current_train_loaders_idx_passed = list(np.where(passed_mask)[0])
        down_weight_factors = torch.ones_like(train_dataset_full.class_weights)
        for i in current_train_loaders_idx_passed:
            down_weight_factors[train_dataset_full.chunks_classes[i]] = args.dw_factor
        print('exit_thresh: {}'.format(exit_thresh))
        print(len(current_train_loaders_idx_passed))
        print(torch.sum(down_weight_factors))
    elif args.exit_strategy == 'dropout':
        current_train_loaders_idx = list(np.where(np.array(prec3_per_chunk) < exit_thresh)[0])
        current_train_datasets = []
        for i in current_train_loaders_idx:
            current_train_datasets.append(train_datasets[i])
        print(len(current_train_datasets))
        train_dataset = ConcatDataset(current_train_datasets)
    else:
        raise RuntimeError('Invalid exit strategy: {}'.format(args.exit_strategy))

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

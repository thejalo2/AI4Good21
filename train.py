import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
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

from utils import Params, AverageMeter, accuracy, save_checkpoint, train_epoch, validate
import data
from models import SharedEmbedderModel


args = Params()
best_prec3 = 0.0
cudnn.benchmark = True

# model
model = SharedEmbedderModel(num_classes=8142, hidden_size=768).cuda()
model.alpha = 1.0
model.inference_alpha = args.inference_alpha

# data
config = resolve_data_config({}, model=model.embedder)
train_dataset = data.INAT(args.data_root, args.train_file, args.cat_file, config, is_train=True, double_img=args.resampling)
val_dataset = data.INAT(args.data_root, args.val_file, args.cat_file, config, is_train=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

# loss & optimizer
criterion = nn.CrossEntropyLoss().cuda()
criterion_reweighted = nn.CrossEntropyLoss(weight=train_dataset.class_weights).cuda()
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
for epoch in range(args.start_epoch, args.epochs):
    model.alpha = 1 - (epoch / args.epochs) ** 2
    # model.alpha = 0.5
    if args.reweighting:
        train_epoch(args, train_loader, model, criterion, optimizer, epoch, criterion_reweighted)
    else:
        train_epoch(args, train_loader, model, criterion, optimizer, epoch)
    prec3 = validate(args, val_loader, model, criterion, False)

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











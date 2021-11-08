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
import inat2018_loader


args = Params()
best_prec3 = 0.0
cudnn.benchmark = True

# model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
model = model.cuda()

# data
config = resolve_data_config({}, model=model)
train_dataset = inat2018_loader.INAT(args.data_root, args.train_file, args.cat_file, config, is_train=True)
val_dataset = inat2018_loader.INAT(args.data_root, args.val_file, args.cat_file, config, is_train=False)
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
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('...... done loading checkpoint {}'.format(args.resume))
    else:
        print('... no checkpoint found at {}'.format(args.resume))

# training loop
for epoch in range(args.start_epoch, args.epochs):
    train_epoch(args, train_loader, model, criterion, optimizer, epoch)

    if args.reweighting :
        train_epoch(args, train_loader, model, criterion, optimizer, epoch, criterion_reweighted)
    else : 
        train_epoch(args, train_loader, model, criterion, optimizer, epoch)

    prec3 = validate(args, val_loader, model, criterion, False)

    # save model
    is_best = prec3 > best_prec3
    best_prec3 = max(prec3, best_prec3)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec3': best_prec3,
        'optimizer': optimizer.state_dict(),
    }, is_best, args.save_path)











# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models

import inat2018_loader

import utils
from params import Params


best_prec3 = 0.0
args = Params()

# load pretrained model
model = models.wide_resnet50_2(pretrained=True)
model.fc = nn.Linear(2048, args.num_classes)
model.aux_logits = False
model = model.cuda()

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec3 = checkpoint['best_prec3']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

# data loading code
train_dataset = inat2018_loader.INAT(args.data_root, args.train_file, args.cat_file, is_train=True)
val_dataset = inat2018_loader.INAT(args.data_root, args.val_file, args.cat_file, is_train=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

for epoch in range(args.start_epoch, args.epochs):
    # adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    utils.train(args, train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    prec3 = utils.validate(args, val_loader, model, criterion, False)

    # remember best prec@1 and save checkpoint
    is_best = prec3 > best_prec3
    best_prec3 = max(prec3, best_prec3)
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec3': best_prec3,
        'optimizer': optimizer.state_dict(),
    }, is_best, args.save_path)

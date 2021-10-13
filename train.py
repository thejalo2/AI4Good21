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

from utils import Params, AverageMeter, accuracy, save_checkpoint
import inat2018_loader


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print('Epoch:{0}'.format(epoch))
    print('Itr\t\tTime\t\tData\t\tLoss\t\tPrec@1\t\tPrec@3')

    for i, (im, im_id, target, tax_ids) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # push to GPU & predict
        im, target = im.cuda(), target.cuda()
        input_var = torch.autograd.Variable(im)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        loss = criterion(output, target_var)
        losses.update(loss.item(), im.size(0))

        # compute gradients and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy
        if i % args.acc_freq == 0:
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            top1.update(prec1[0], im.size(0))
            top3.update(prec3[0], im.size(0))

        # print timings and loss
        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  '{data_time.val:.2f} ({data_time.avg:.2f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})\t'
                  '{top1.val:.2f} ({top1.avg:.2f})\t'
                  '{top3.val:.2f} ({top3.avg:.2f})'
                  .format(i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1,
                          top3=top3), flush=True)


def validate(val_loader, model, criterion, save_preds=False):

    with torch.inference_mode():

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        pred = []
        im_ids = []

        print('Validate:\tTime\t\tLoss\t\tPrec@1\t\tPrec@3')
        for i, (im, im_id, target, tax_ids) in enumerate(val_loader):

            im = im.cuda()
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(im)
            target_var = torch.autograd.Variable(target)

            # predict
            output = model(input_var)
            loss = criterion(output, target_var)

            # store the top K classes for the prediction
            if save_preds:
                im_ids.append(im_id.cpu().numpy().astype(np.int))
                _, pred_inds = output.data.topk(3, 1, True, True)
                pred.append(pred_inds.cpu().numpy().astype(np.int))

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.item(), im.size(0))
            top1.update(prec1[0], im.size(0))
            top3.update(prec3[0], im.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('[{0}/{1}]\t'
                      '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                      '{loss.val:.3f} ({loss.avg:.3f})\t'
                      '{top1.val:.2f} ({top1.avg:.2f})\t'
                      '{top3.val:.2f} ({top3.avg:.2f})'
                      .format(i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top3=top3))

        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(top1=top1, top3=top3))

        if save_preds:
            return top3.avg, np.vstack(pred), np.hstack(im_ids)
        else:
            return top3.avg


args = Params()
best_prec3 = 0.0
cudnn.benchmark = True

# model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
model = model.cuda()
# model = models.resnet34(pretrained=True)
# model.fc = nn.Linear(512, args.num_classes)
# model.aux_logits = False
# model = model.cuda()

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
    train_epoch(train_loader, model, criterion, optimizer, epoch)
    prec3 = validate(val_loader, model, criterion, False)

    # save model
    is_best = prec3 > best_prec3
    best_prec3 = max(prec3, best_prec3)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec3': best_prec3,
        'optimizer': optimizer.state_dict(),
    }, is_best, args.save_path)











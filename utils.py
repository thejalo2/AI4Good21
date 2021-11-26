import shutil
import torch
import os
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Params:
    # set to model path to continue training
    # resume = 'checkpoints/first_reweight/reproduce/reproduce_best.pth.tar'
    # resume = 'exit.pth.tar'
    resume = ''

    # paths
    if os.name == 'nt':
        data_root = 'C:/Users/janik/inat_data/'
    else:
        data_root = '/cluster/scratch/ljanik/inat_data/'
    train_file = data_root + 'train2018.json'
    val_file = data_root + 'val2018.json'
    cat_file = data_root + 'categories.json'
    save_path = 'exit.pth.tar'

    # hyper-parameters
    num_classes = 8142
    if os.name == 'nt':
        batch_size = 8
    else:
        batch_size = 16
    lr = 1e-5
    epochs = 50
    start_epoch = 0
    start_alpha = 1.0
    inference_alpha = 0.0
    share_embedder = True
    use_ldam = False
    reweighting = True
    resampling = False
    combine_logits = False
    merged_training = False
    beta = None
    chunk_size = 815
    exit_thresh = 80

    # system variables
    print_freq = 100
    acc_freq = 1
    if os.name == 'nt':
        workers = 0
    else:
        workers = 4


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LDAMLoss(nn.Module):
    """
    https://github.com/kaidic/LDAM-DRW/blob/3193f05c1e6e8c4798c5419e97c5a479d991e3e9/losses.py#L23
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, filename.replace('.pth.tar', '_best.pth.tar'))


def train_epoch(args, train_loader, model, criterion, optimizer, epoch, criterion_reweighted=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {'cb': AverageMeter(), 'rb': AverageMeter(), 'total': AverageMeter()}
    top1 = {'cb': AverageMeter(), 'rb': AverageMeter()}
    top3 = {'cb': AverageMeter(), 'rb': AverageMeter()}

    # switch to train mode
    model.train()

    end = time.time()
    print('Epoch:{0}'.format(epoch))
    print('Itr'.ljust(15), 'Time'.ljust(15), 'Data'.ljust(15), 'Loss_total'.ljust(20), 'Loss_1'.ljust(20),
          'Loss_2'.ljust(20), 'Prec@1_1'.ljust(20), 'Prec@3_1'.ljust(20), 'Prec@1_2'.ljust(20), 'Prec@3_2'.ljust(20))

    for i, sample in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if len(sample) > 4:
            (im_1, im_id_1, target_1, tax_ids_1, im_2, im_id_2, target_2, tax_ids_2) = sample

            im_1, target_1 = im_1.cuda(), target_1.cuda()
            input_var_1 = torch.autograd.Variable(im_1)
            target_var_1 = torch.autograd.Variable(target_1)

            im_2, target_2 = im_2.cuda(), target_2.cuda()
            input_var_2 = torch.autograd.Variable(im_2)
            target_var_2 = torch.autograd.Variable(target_2)

            # predict
            output_1, output_2 = model(input_var_1, input_var_2)

        else:
            (im_1, im_id_1, target_1, tax_ids_1) = sample

            im_1, target_1 = im_1.cuda(), target_1.cuda()
            input_var_1 = torch.autograd.Variable(im_1)
            target_var_1 = torch.autograd.Variable(target_1)

            target_2 = target_1
            im_2 = im_1
            target_var_2 = target_var_1

            # predict
            if args.combine_logits:
                output_1 = model(input_var_1, None, combine=True)
                output_2 = output_1
            else:
                output_1, output_2 = model(input_var_1, None, combine=False)

        loss_1 = criterion(output_1, target_var_1)
        if criterion_reweighted is None:
            loss_2 = criterion(output_2, target_var_2)
        else:
            loss_2 = criterion_reweighted(output_2, target_var_2)

        loss = model.alpha * loss_1 + (1 - model.alpha) * loss_2

        if args.merged_training:
            # note: this only makes sense if the two images are the same
            output_merged = model.alpha * output_1 + (1 - model.alpha) * output_2
            loss_3 = criterion(output_merged, target_var_1)
            if criterion_reweighted is None:
                loss_4 = criterion(output_merged, target_var_2)
            else:
                loss_4 = criterion_reweighted(output_merged, target_var_2)
            beta = 0.5
            loss = beta * loss + (1. - beta) * (model.alpha * loss_3 + (1 - model.alpha) * loss_4)

        losses['cb'].update(loss_1.item(), im_1.size(0))
        losses['rb'].update(loss_2.item(), im_1.size(0))
        losses['total'].update(loss.item(), im_1.size(0))

        # compute gradients and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy
        if i % args.acc_freq == 0:
            prec1_1, prec3_1 = accuracy(output_1.data, target_1, topk=(1, 3))
            top1['cb'].update(prec1_1[0], im_1.size(0))
            top3['cb'].update(prec3_1[0], im_1.size(0))

            prec1_2, prec3_2 = accuracy(output_2.data, target_2, topk=(1, 3))
            top1['rb'].update(prec1_2[0], im_2.size(0))
            top3['rb'].update(prec3_2[0], im_2.size(0))

        # print timings and loss
        if i % args.print_freq == 0:
            print('[{0}/{1}]'.format(i, len(train_loader)).ljust(15),
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})'.format(batch_time=batch_time).ljust(15),
                  '{data_time.val:.2f} ({data_time.avg:.2f})'.format(data_time=data_time).ljust(15),
                  '{loss.val:.3f} ({loss.avg:.3f})'.format(loss=losses['total']).ljust(20),
                  '{loss_1.val:.3f} ({loss_1.avg:.3f})'.format(loss_1=losses['cb']).ljust(20),
                  '{loss_2.val:.3f} ({loss_2.avg:.3f})'.format(loss_2=losses['rb']).ljust(20),
                  '{top1_1.val:.2f} ({top1_1.avg:.2f})'.format(top1_1=top1['cb']).ljust(20),
                  '{top3_1.val:.2f} ({top3_1.avg:.2f})'.format(top3_1=top3['cb']).ljust(20),
                  '{top1_2.val:.2f} ({top1_2.avg:.2f})'.format(top1_2=top1['rb']).ljust(20),
                  '{top3_2.val:.2f} ({top3_2.avg:.2f})'.format(top3_2=top3['rb']).ljust(20),
                  flush=True)


def validate(args, val_loader, model, criterion, save_preds=False):
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
            output = model(input_var, None)
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
            return top1.avg, top3.avg, np.vstack(pred), np.hstack(im_ids)
        else:
            return top1.avg, top3.avg


def get_chunk_idx(chunks_classes, classes_inv):
    num_chunks = len(chunks_classes)
    # stores which images (index) belong to each chunk
    chunks_imgs = [[] for _ in range(num_chunks)]
    for chunk_idx, chunk in enumerate(chunks_classes):
        for species_id in chunk:
            chunks_imgs[chunk_idx] += classes_inv[species_id]
    return chunks_imgs

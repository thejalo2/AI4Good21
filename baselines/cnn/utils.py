import torch
import time
import shutil
import numpy as np

CUDA_THINGS = (torch.Tensor, torch.nn.utils.rnn.PackedSequence)


# push things to a device
def things2dev(obj, dev):
    if isinstance(obj, CUDA_THINGS):
        return obj.to(dev)
    if isinstance(obj, list):
        return [things2dev(x, dev) for x in obj]
    if isinstance(obj, tuple):
        return tuple(things2dev(list(obj), dev))
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = things2dev(v, dev)
    return obj


def save_model(model, save_path):
    sd = model.state_dict()
    sd = things2dev(sd, 'cpu')
    torch.save(sd, save_path)
    print('Saved model to: %s' % save_path)


def train(args, train_loader, model, criterion, optimizer, epoch):
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
    for i, (input, im_id, target, tax_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # record loss
        losses.update(loss.item(), input.size(0))

        # measure accuracy
        if i % args.acc_freq == 0:
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            top1.update(prec1[0], input.size(0))
            top3.update(prec3[0], input.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  '{data_time.val:.2f} ({data_time.avg:.2f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})\t'
                  '{top1.val:.2f} ({top1.avg:.2f})\t'
                  '{top3.val:.2f} ({top3.avg:.2f})'.format(
                i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3), flush=True)


def validate(args, val_loader, model, criterion, save_preds=False):
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
    for i, (input, im_id, target, tax_ids) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        with torch.inference_mode():
            output = model(input_var)
        loss = criterion(output, target_var)

        if save_preds:
            # store the top K classes for the prediction
            im_ids.append(im_id.cpu().numpy().astype(np.int))
            _, pred_inds = output.data.topk(3, 1, True, True)
            pred.append(pred_inds.cpu().numpy().astype(np.int))

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  '{batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  '{loss.val:.3f} ({loss.avg:.3f})\t'
                  '{top1.val:.2f} ({top1.avg:.2f})\t'
                  '{top3.val:.2f} ({top3.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    if save_preds:
        return top3.avg, np.vstack(pred), np.hstack(im_ids)
    else:
        return top3.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        print("\tSaving new best model")
        shutil.copyfile(filename, filename.replace('.pth.tar', '_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
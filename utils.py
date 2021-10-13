import os
import shutil
import torch


class Params:

    # set to model path to continue training
    resume = ''

    # paths
    if os.name == 'nt':
        data_root = 'C:/Users/janik/inat_data/'
    else:
        data_root = '/cluster/scratch/ljanik/inat_data/'
    train_file = data_root + 'train2018.json'
    val_file = data_root + 'val2018.json'
    cat_file = data_root + 'categories.json'
    save_path = 'vit_base_patch16_224.pth.tar'

    # hyper-parameters
    num_classes = 8142
    batch_size = 16
    lr = 1e-5
    epochs = 100
    start_epoch = 0

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


# def accuracy(output, target, topk=(1,)):
#     """
#     Computes the accuracy over the k top predictions for the specified values of k
#
#     Copyright 2020 Ross Wightman
#     https://github.com/rwightman/pytorch-image-models/blob/02daf2ab943ce2c1646c4af65026114facf4eb22/timm/utils/metrics.py
#     """
#     maxk = min(max(topk), output.size()[1])
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.reshape(1, -1).expand_as(pred))
#     return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


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
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models

import inat2018_loader

import utils


class Params:
    # hyper-parameters
    num_classes = 8142
    workers = 4
    epochs = 100
    start_epoch = 0
    batch_size = 16
    lr = 1e-5
    lr_decay = 0.94
    epoch_decay = 4
    momentum = 0.9
    weight_decay = 1e-4
    print_freq = 100
    acc_freq = 1

    # paths
    resume = 'resnet_baseline_training/adam_wide_resnet_50_best.pth.tar'  # set this to path of model to resume training
    save_path = 'adam_wide_resnet_50.pth.tar'  # set this to path of where to save the model
    data_root = 'C:/Users/janik/inat_data/'
    train_file = data_root + 'train2018.json'
    val_file = data_root + 'val2018.json'
    cat_file = data_root + 'categories.json'

    # set evaluate to True to run the test set
    evaluate = True
    save_preds = True
    op_file_name = 'inat2018_test_preds.csv'  # submission file
    if evaluate:
        val_file = data_root + 'val2018.json'
        # val_file = data_root + 'test2018.json'


args = Params()
cudnn.benchmark = True

# model
model = models.wide_resnet50_2(pretrained=True)
# model = models.resnet34(pretrained=True)
model.fc = nn.Linear(2048, args.num_classes)
# model.fc = nn.Linear(512, args.num_classes)
model.aux_logits = False
model = model.cuda()
model.eval()

# load model
assert args.resume
print("=> loading checkpoint '{}'".format(args.resume))
checkpoint = torch.load(args.resume)
args.start_epoch = checkpoint['epoch']
best_prec3 = checkpoint['best_prec3']
model.load_state_dict(checkpoint['state_dict'])

# data
train_dataset = inat2018_loader.INAT(args.data_root, args.train_file, args.cat_file, is_train=True)
val_dataset = inat2018_loader.INAT(args.data_root, args.val_file, args.cat_file, is_train=False)
val_loader_sep = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=0, pin_memory=True)

# validate per class
accs = [{'prec1': 0, 'prec3': 0, 'count': 0} for _ in range(len(train_dataset.ord_lookup))]
for i, (input, im_id, target, tax_ids) in enumerate(val_loader_sep):
    input = input.cuda()
    target = target.cuda(non_blocking=True)
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)
    with torch.inference_mode():
        output = model(input_var)

    prec1, prec3 = utils.accuracy(output.data, target, topk=(1, 3))
    idx = train_dataset.ord_lookup[target[0].item()]
    accs[idx]['prec1'] += prec1.item()
    accs[idx]['prec3'] += prec3.item()
    accs[idx]['count'] += 1

    if i % args.print_freq == 0:
        print('[{0}/{1}]'.format(i, len(val_loader_sep)))

accs_avg_prec1 = [e['prec1'] / e['count'] for e in accs]
accs_avg_prec3 = [e['prec3'] / e['count'] for e in accs]

# plot
s = 1
k = 1
accs_avg_prec3_acc = [sum(accs_avg_prec3[i:i + s]) / s for i in range(0, len(accs_avg_prec3) - (s-1), k)]
# plt.bar(k*np.arange(len(accs_avg_prec3_acc)), accs_avg_prec3_acc, width=s, align='edge')
# plt.plot(k*np.arange(len(accs_avg_prec3_acc)), accs_avg_prec3_acc, 'r', alpha=0.1)
plt.plot(k*np.arange(len(accs_avg_prec3_acc)), gaussian_filter1d(accs_avg_prec3_acc, 300), 'r')
c = train_dataset.counts
counts = (c - np.min(c)) / (np.max(c) - np.min(c)) * (90 - 50) + 50
plt.bar(range(len(counts)), counts, width=1, alpha=0.5, align='edge')
plt.ylim([50, 90])
plt.xlabel('Species')
plt.ylabel('accuracy %')

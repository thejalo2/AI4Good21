import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from scipy.ndimage import gaussian_filter1d

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from PIL import Image
import urllib
import os
import os.path as osp
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import Params, AverageMeter, accuracy, save_checkpoint, train_epoch, validate
import inat2018_loader

args = Params()
cudnn.benchmark = True

# model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
model = model.cuda()

# data
config = resolve_data_config({}, model=model)
val_dataset = inat2018_loader.INAT(args.data_root, args.val_file, args.cat_file, config, is_train=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

# load model
assert args.resume
print('... loading checkpoint {}'.format(args.resume))
checkpoint = torch.load(args.resume)
args.start_epoch = checkpoint['epoch']
best_prec3 = checkpoint['best_prec3']
model.load_state_dict(checkpoint['state_dict'])
print('...... done loading checkpoint {}'.format(args.resume))

# validate
criterion = nn.CrossEntropyLoss().cuda()
prec3 = validate(args, val_loader, model, criterion, False)

# validate per class
train_dataset = inat2018_loader.INAT(args.data_root, args.train_file, args.cat_file, config, is_train=True)
val_loader_sep = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
accs = [{'prec1': 0, 'prec3': 0, 'count': 0} for _ in range(len(train_dataset.ord_lookup))]
for i, (im, im_id, target, tax_ids) in enumerate(val_loader_sep):
    with torch.inference_mode():
        im = im.cuda()
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(im)
        output = model(input_var)
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        idx = train_dataset.ord_lookup[target[0].item()]
        accs[idx]['prec1'] += prec1.item()
        accs[idx]['prec3'] += prec3.item()
        accs[idx]['count'] += 1

        if i % args.print_freq == 0 and i > 0:
            print('[{0}/{1}]'.format(i, len(val_loader_sep)), flush=True)

accs_avg_prec1 = [e['prec1'] / e['count'] for e in accs]
accs_avg_prec3 = [e['prec3'] / e['count'] for e in accs]

# plot
s = 1
k = 1
accs_avg_prec3_acc = [sum(accs_avg_prec3[i:i + s]) / s for i in range(0, len(accs_avg_prec3) - (s-1), k)]
print(accs_avg_prec3_acc)
# plt.bar(k*np.arange(len(accs_avg_prec3_acc)), accs_avg_prec3_acc, width=s, align='edge')
plt.plot(k*np.arange(len(accs_avg_prec3_acc)), accs_avg_prec3_acc, 'r', alpha=0.1)
plt.plot(k*np.arange(len(accs_avg_prec3_acc)), gaussian_filter1d(accs_avg_prec3_acc, 300), 'r')
c = train_dataset.counts
counts = (c - np.min(c)) / (np.max(c) - np.min(c)) * (90 - 50) + 50
plt.bar(range(len(counts)), counts, width=1, alpha=0.5, align='edge')
plt.ylim([50, 100])
plt.xlabel('Species')
plt.ylabel('accuracy %')
plt.show()
plt.savefig('fig.png')

plt.close()
counts = [e['count'] for e in accs]
plt.bar(range(len(counts)), counts, width=1)
plt.savefig('fig2.png')


# train_loader_sep = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
#                                                num_workers=0, pin_memory=True)
# accs = [{'prec1': 0, 'prec3': 0, 'count': 0} for _ in range(len(train_dataset.ord_lookup))]
# for i, (_, _, target, _) in enumerate(train_loader_sep):
#     idx = train_dataset.ord_lookup[target[0].item()]
#     accs[idx]['count'] += 1
#     if i % 100 == 0:
#         print(i, ' / ', len(train_loader_sep))
# counts = [e['count'] for e in accs]
# plt.bar(range(len(counts)), counts, width=1)

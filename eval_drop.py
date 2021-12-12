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
import argparse

from utils import Params, AverageMeter, accuracy, save_checkpoint, train_epoch, validate
import data
from models import SharedEmbedderModel
import pickle
import json

args = Params()
cudnn.benchmark = True

# For the per-category accuracy 
parser = argparse.ArgumentParser()
parser.add_argument("-cat", type=str, default='supercategory',
                    help='Over which category to plot the validation accuracies.')
parser.add_argument("-avg", type=int, default=0, help='Get the per-category average validation accuracy.')
parsed_args = parser.parse_args()

# model
model = SharedEmbedderModel(num_classes=8142, hidden_size=768, share_embedder=args.share_embedder).cuda()
model.inference_alpha = args.inference_alpha

# data
try:
    config = resolve_data_config({}, model=model.embedder)
except:
    config = resolve_data_config({}, model=model.embedder_rb)
val_dataset = data.INAT(args.data_root, args.val_file, args.cat_file, config, args.beta, is_train=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

# load model
assert args.resume
print('... loading checkpoint {}'.format(args.resume))
checkpoint = torch.load(args.resume)
args.start_epoch = checkpoint['epoch']
best_prec3 = checkpoint['best_prec3']
model.load_state_dict(checkpoint['state_dict'])
print('...... done loading checkpoint {}, epoch {}'.format(args.resume, args.start_epoch))

# validate
criterion = nn.CrossEntropyLoss().cuda()
prec1_overall, prec3_overall = validate(args, val_loader, model, criterion, False)

#####################################
# Set-up :

with open(args.val_file) as data_file: ann_data = json.load(data_file)
with open(args.cat_file) as cat_data_file: cat_data = json.load(cat_data_file)

# All the categories in the consideed `category` (eg. supercategory, or kindgom)
all_of_category = set()
for cat in cat_data:
    all_of_category.add(cat[parsed_args.cat])

# Per-category average validation accuracy
per_cat_acc = {k: [] for k in list(all_of_category)}

# Per-category per-species average validation accuracy
per_cat_avg_acc = {k: [] for k in list(all_of_category)}
per_cat_counts = {k: [] for k in list(all_of_category)}
per_cat_colors = {k: [] for k in list(all_of_category)}

#####################################
# Validate per class

train_dataset = data.INAT(args.data_root, args.train_file, args.cat_file, config, args.beta, is_train=True)
val_loader_sep = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
accs = [{'prec1': 0, 'prec3': 0, 'count': 0} for _ in range(len(train_dataset.ord_lookup))]
for i, (im, im_id, target, tax_ids) in enumerate(val_loader_sep):
    with torch.inference_mode():
        im = im.cuda()
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(im)
        output = model(input_var, None)
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        idx = train_dataset.ord_lookup[target[0].item()]
        accs[idx]['prec1'] += prec1.item()
        accs[idx]['prec3'] += prec3.item()
        accs[idx]['count'] += 1

        cat_id = ann_data['annotations'][i]['category_id']
        category = cat_data[cat_id][parsed_args.cat]
        per_cat_acc[category].append(prec3.item())

        if i % args.print_freq == 0 and i > 0:
            print('[{0}/{1}]'.format(i, len(val_loader_sep)), flush=True)

if parsed_args.avg:
    for category, accuracies in per_cat_acc.items():
        print()
        print('Average validation scores per {}'.format(parsed_args.cat))
        print('{} : {:02n}'.format(category, np.mean(accuracies)))
        print()

accs_avg_prec1 = [e['prec1'] / e['count'] for e in accs]
accs_avg_prec3 = [e['prec3'] / e['count'] for e in accs]

#####################################
# The plotting

s, k = 1, 1
accs_avg_prec3_acc = [sum(accs_avg_prec3[i:i + s]) / s for i in range(0, len(accs_avg_prec3) - (s - 1), k)]
c = train_dataset.counts_ordered
counts = (c - np.min(c)) / (np.max(c) - np.min(c)) * (90 - 50) + 50

for rank, acc in enumerate(accs_avg_prec3_acc):
    cat_id = train_dataset.ord_lookup_inv[rank]
    category = ann_data['categories'][cat_id][parsed_args.cat]
    per_cat_avg_acc[category].append(acc)
    per_cat_counts[category].append(counts[rank])
    if 1500 <= rank <= 2500:
        per_cat_colors[category].append('red')
    else:
        per_cat_colors[category].append('blue')

if not osp.isdir(parsed_args.cat): os.mkdir(parsed_args.cat)

for category in list(all_of_category):
    num_species = len(per_cat_avg_acc[category])
    len_counts = len(per_cat_counts[category])

    plt.figure()
    plt.plot(k * np.arange(num_species), gaussian_filter1d(per_cat_avg_acc[category], max(1, num_species // 25)), 'r')
    plt.bar(range(len_counts), per_cat_counts[category], width=1, alpha=0.5, align='edge',
            color=per_cat_colors[category])
    plt.ylim([0, 100])
    plt.xlabel('Species')
    plt.ylabel('accuracy %')
    plt.title('Avg val acc for {} : {}'.format(parsed_args.cat, category))
    plt.show()
    plt.savefig('{}/{}-fig.png'.format(parsed_args.cat, category))

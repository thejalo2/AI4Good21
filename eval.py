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
parser.add_argument("-cat",    type = str)
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
# Accuracy plot per supercategory :

if parsed_args.cat : 
    # need access to ann_data and cat_data : 
    with open(args.val_file) as data_file: ann_data = json.load(data_file)
    with open(args.cat_file) as cat_data_file: cat_data = json.load(cat_data_file)

    all_of_category = set()

    for cat in cat_data :
        all_of_category.add(cat[parsed_args.cat])

    per_cat_acc = {k: [] for k in list(all_of_category)}
    per_cat_avg_acc ={k: [] for k in list(all_of_category)}

#####################################


# validate per class
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

        if parsed_args.cat :
            cat_id = ann_data['annotations'][i]['category_id']
            category = cat_data[cat_id][parsed_args.cat]
            per_cat_acc[category].append(prec3.item())

        if i % args.print_freq == 0 and i > 0:
            print('[{0}/{1}]'.format(i, len(val_loader_sep)), flush=True)

""" uncomment to get the avg validation accuracy per category
if parsed_args.cat : 
    for category, accuracies in per_cat_acc.items() :
        print()
        print('Average validation scores per {}'.format(parsed_args.cat)) 
        print('{} : {:02n}'.format(category, np.mean(accuracies)))
        print()
"""

accs_avg_prec1 = [e['prec1'] / e['count'] for e in accs]
accs_avg_prec3 = [e['prec3'] / e['count'] for e in accs]

# plot
s = 1
k = 1
accs_avg_prec3_acc = [sum(accs_avg_prec3[i:i + s]) / s for i in range(0, len(accs_avg_prec3) - (s-1), k)]
print(accs_avg_prec3_acc)
# plt.bar(k*np.arange(len(accs_avg_prec3_acc)), accs_avg_prec3_acc, width=s, align='edge')
# plt.plot(k*np.arange(len(accs_avg_prec3_acc)), accs_avg_prec3_acc, 'r', alpha=0.1)
plt.figure()
plt.plot(k*np.arange(len(accs_avg_prec3_acc)), gaussian_filter1d(accs_avg_prec3_acc, 300), 'r')
c = train_dataset.counts_ordered
counts = (c - np.min(c)) / (np.max(c) - np.min(c)) * (90 - 50) + 50
plt.bar(range(len(counts)), counts, width=1, alpha=0.5, align='edge')
plt.ylim([50, 100])
plt.xlabel('Species')
plt.ylabel('accuracy %')
plt.title('Top-1: {}%, Top-3: {}%'.format(round(prec1_overall.item(), 3), round(prec3_overall.item(), 3)))
plt.show()
plt.savefig('fig.png')


#####################################
# Accuracy plot per supercategory :

if parsed_args.cat :

    for rank, acc in enumerate(accs_avg_prec3_acc) :
        cat_id = train_dataset.ord_lookup_inv[rank] 
        category = ann_data['categories'][cat_id][parsed_args.cat]
        per_cat_avg_acc[category].append(acc)

    if not osp.isdir(parsed_args.cat) : os.mkdir(parsed_args.cat)

    for category in list(all_of_category) :

        num_species = len(per_cat_avg_acc[category])

        s, k = 1, 1
        plt.figure()
        plt.plot(k * np.arange(num_species),gaussian_filter1d(per_cat_avg_acc[category], max(1, num_species//25)), 'r')
        #plt.bar(range(len(counts)), counts, width=1, alpha=0.5, align='edge')
        plt.ylim([0, 100])
        plt.xlabel('Species')
        plt.ylabel('accuracy %')
        plt.title('Avg val acc for {} : {}'.format(parsed_args.cat,category))
        plt.show()
        plt.savefig('{}/{}-fig.png'.format(parsed_args.cat, category))

#####################################




# with open('a=0.0.pkl', 'wb') as f:
#     pickle.dump(accs_avg_prec3_acc, f)

# plt.close()
# counts = [e['count'] for e in accs]
# plt.bar(range(len(counts)), counts, width=1)
# plt.savefig('fig2.png')


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

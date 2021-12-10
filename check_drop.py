from PIL import Image
import os
import os.path as osp
import json
import random
import numpy as np
import argparse
from utils import Params
import matplotlib.pyplot as plt
import collections

args_util = Params()

parser = argparse.ArgumentParser()
parser.add_argument("-low", type=int, default=1500)
parser.add_argument("-high", type=int, default=2500)
parser.add_argument("-root", type=str)
parser.add_argument("-ann", type=str)
parser.add_argument("-cat", type=str, default='supercategory')
args = parser.parse_args()

if not (args.root and args.ann and args.cat):
    data_root = args_util.data_root
    ann_file, cat_file = data_root + 'train2018.json', data_root + 'categories.json'

# load annotations
print('Loading annotations from: ' + os.path.basename(ann_file))
with open(ann_file) as data_file:
    ann_data = json.load(data_file)

# load un-obfuscated category data
with open(cat_file) as cat_data_file:
    cat_data = json.load(cat_data_file)
ann_data['categories'] = cat_data

classes = {}
cats = ['supercategory', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'name']
for c in cats:
    classes[c] = []
for e in cat_data:
    for c in cats:
        classes[c].append(e[c])

res = dict.fromkeys([e['category_id'] for e in ann_data['annotations']], 0)
for e in ann_data['annotations']:
    res[int(e['category_id'])] += 1

els = list(res.items())
my_list = els
my_list.sort(key=lambda i: i[1], reverse=True)

cats_all = []
for cat_id, count in my_list:
    cats_all.append(cat_data[cat_id][args.cat])
# hist_dict_all = collections.Counter(np.array(cats_all).flatten())
cats_all = list(set(cats_all))

out_path = r'drop_analysis'
os.makedirs(out_path, exist_ok=True)
frames = 500
span = 500
highs = np.linspace(span, 8142, frames).astype(int)
for i, high in enumerate(highs):
    low = high - span

    cats = []
    # low = args.low
    # high = args.high
    category = args.cat

    for cat_id, count in my_list[low:high]:
        cats.append(cat_data[cat_id][category])
    hist_dict = collections.Counter(np.array(cats).flatten())
    for c in cats_all:
        if c not in hist_dict:
            hist_dict[c] = 0
    hist_dict = collections.OrderedDict(sorted(hist_dict.items()))

    print(hist_dict)

    plt.figure(figsize=(15, 10))
    plt.bar(x=hist_dict.keys(), height=hist_dict.values())
    plt.ylim(0, 250)
    plt.xlabel('category')
    plt.ylabel('number of instances of this category in the range')
    plt.title('range: [{} , {}] center: {}'.format(low, high, low + (high - low) // 2))
    plt.savefig(osp.join(out_path, str(i) + '.png'))
    # plt.show()
    plt.close()

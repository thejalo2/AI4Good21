from PIL import Image
import os
import json
import random
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-low",       type = int, default=1500)
parser.add_argument("-high",       type = int, default=2500)
parser.add_argument("-root",    type = str)
parser.add_argument("-ann",    type = str)
parser.add_argument("-cat",    type = str, default = 'supercategory')
args = parser.parse_args()

if not (args.root and args.ann and args.cat) : 
    data_root  = '../../../scratch/gsialelli/inat_data/'
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
my_list.sort(key=lambda i:i[1],reverse=True)

cats = []
low = args.low
high = args.high
category = args.cat

for cat_id,count in my_list[low:high] : cats.append(cat_data[cat_id][category])

import collections

m = np.array(cats)

print(collections.Counter(m.flatten()))







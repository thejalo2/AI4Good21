import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np
import matplotlib.pyplot as plt


def default_loader(path):
    return Image.open(path).convert('RGB')


def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0] * len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic


class INAT(data.Dataset):
    def __init__(self, root, ann_file, cat_file, config, is_train=True):

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # load un-obfuscated category data
        with open(cat_file) as cat_data_file:
            cat_data = json.load(cat_data_file)
        ann_data['categories'] = cat_data

        # category exploration
        classes = {}
        cats = ['supercategory', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'name']
        for c in cats:
            classes[c] = []
        for e in cat_data:
            for c in cats:
                classes[c].append(e[c])

        # print((classes['name']))
        print((classes['name'][6829]))
        # for c in cats:
        #     print(c)
        #     print(set(classes[c]))
        #     print()

        # data distribution exploration
        # print(ann_data['categories'][ann_data['annotations'][0]['category_id']])
        if 'annotations' in ann_data.keys():
            counts = [0 for _ in range(len(ann_data['categories']))]
            for e in ann_data['annotations']:
                counts[int(e['category_id'])] += 1
            counts = np.array(counts)
            ord = np.argsort(counts)[::-1]
            print(ord)
            self.ord_lookup = {ord[i]: i for i in ord}
            counts = counts[ord]
            self.counts = counts
        # plt.bar(range(len(ann_data['categories'])), counts, width=1)#, log=True)
        # plt.xlabel('Species')
        # plt.ylabel('Number of images')

        # c = 'supercategory'
        # counts = {c: 0 for c in classes[c]}
        # for e in ann_data['annotations']:
        #     counts[ann_data['categories'][int(e['category_id'])][c]] += 1
        # counts = sorted([counts[c] for c in counts.keys()], reverse=True)
        # print(max(counts))
        # plt.bar(range(len(counts)), counts, width=1)#, log=True)
        # plt.xlabel('Superclass')
        # plt.ylabel('Number of images')

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0] * len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
        # 8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # taxonomy: for every element in the tax, mapping id to to the tax element instance
        # classes_taxonomic: for every id, a list of length 7 with the full tax.

        # print out some stats
        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(set(self.classes))) + ' classes')

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # augmentation params
        self.im_size = config['input_size'][1:]
        self.mu_data = config['mean']
        self.std_data = config['std']
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)
        self.train_trafo = transforms.Compose([
            self.scale_aug,
            self.flip_aug,
            self.color_aug,
            self.tensor_aug,
            self.norm_aug
        ])
        self.val_trafo = transforms.Compose([
            self.center_crop,
            self.tensor_aug,
            self.norm_aug
        ])

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]

        if self.is_train:
            img = self.train_trafo(img)
        else:
            img = self.val_trafo(img)

        return img, im_id, species_id, tax_ids

    def __len__(self):
        return len(self.imgs)

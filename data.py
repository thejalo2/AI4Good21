import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import collections


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
    def __init__(self, root, ann_file, cat_file, config, beta, double_img=False, is_train=True, chunk_size=None):

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
        self.ann_data = ann_data
        # print((classes['name'][2844]))

        # data distribution exploration
        if 'annotations' in ann_data.keys():
            counts = [0 for _ in range(len(ann_data['categories']))]
            for e in ann_data['annotations']:
                counts[int(e['category_id'])] += 1
            counts = np.array(counts)
            ord = np.argsort(counts)[::-1]
            print(ord)
            self.ord_lookup = {ord[i]: i for i in ord}  # class index -> rank
            self.ord_lookup_inv = {v: k for k, v in self.ord_lookup.items()}  # rank -> class index
            self.counts_lookup = counts
            self.counts_ordered = counts[ord]

        # pre computations for weighted sampling
        if 'annotations' in ann_data.keys():
            max_count = np.max(self.counts_lookup)
            count_normalizer = np.sum(max_count / self.counts_lookup)
            self.class_probas = np.zeros_like(self.counts_lookup, dtype=float)
            for class_index, class_count in enumerate(self.counts_lookup):
                self.class_probas[class_index] = (max_count / class_count) / count_normalizer

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
        self.double_img = double_img
        self.loader = default_loader
        if 'annotations' in ann_data.keys():
            self.num_classes = self.counts_lookup.shape[0]
        self.num_imgs = len(self.imgs)

        if 'annotations' in ann_data.keys():
            if not beta:
                # pre computations for re-weighted loss
                self.class_weights = 1. / self.counts_lookup
                # compensate total weight-down
                # make classes sum to 1
                self.class_weights *= self.num_classes / np.sum(self.class_weights)
                # make training samples sum to 1
                # self.class_weights *= len(self.classes) / np.sum(self.class_weights[self.classes])
                self.class_weights = torch.FloatTensor(self.class_weights).cuda()

            else:
                # https://github.com/kaidic/LDAM-DRW/blob/3193f05c1e6e8c4798c5419e97c5a479d991e3e9/utils.py#L31
                effective_num = 1.0 - np.power(beta, self.counts_lookup)
                self.class_weights = (1.0 - beta) / np.array(effective_num)
                self.class_weights = torch.FloatTensor(self.class_weights).cuda()

            # classes inv stores for a class a list of indices of images which are from that class
            self.classes_inv = [[] for _ in range(self.num_classes)]
            for index in range(self.num_imgs):
                species_id = self.classes[index]
                self.classes_inv[species_id].append(index)

            # split dataset into blocks
            if chunk_size is not None:
                self.chunk_size = chunk_size
                self.ord_lookup_inv_ordered = collections.OrderedDict(sorted(self.ord_lookup_inv.items()))
                self.index_order = list(self.ord_lookup_inv_ordered.values())
                # stores which class (species id) belongs to each chunk
                self.chunks_classes = [self.index_order[i:i + self.chunk_size] for i in
                                       range(0, len(self.index_order), self.chunk_size)]
                self.num_chunks = len(self.chunks_classes)
                # # stores which images (index) belong to each chunk
                # self.chunks_imgs = [[] for _ in range(self.num_chunks)]
                # for chunk_idx, chunk in enumerate(self.chunks_classes):
                #     for species_id in chunk:
                #         self.chunks_imgs[chunk_idx] += self.classes_inv[species_id]

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
        # class_count = self.counts_lookup[species_id]

        if self.is_train:
            img = self.train_trafo(img)
        else:
            img = self.val_trafo(img)

        if self.double_img:
            # sample random image for re-sampling branch
            random_index = np.random.choice(self.num_classes, p=self.class_probas)
            random_path = self.root + self.imgs[random_index]
            random_im_id = self.ids[random_index]
            random_img = self.loader(random_path)
            random_species_id = self.classes[random_index]
            random_tax_ids = self.classes_taxonomic[random_species_id]
            random_class_count = self.counts_lookup[random_species_id]

            if self.is_train:
                random_img = self.train_trafo(random_img)
            else:
                random_img = self.val_trafo(random_img)

            return img, im_id, species_id, tax_ids, random_img, random_im_id, random_species_id, random_tax_ids

        else:
            return img, im_id, species_id, tax_ids

    def __len__(self):
        return len(self.imgs)

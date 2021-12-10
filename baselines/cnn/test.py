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
from params import Params

args = Params()
cudnn.benchmark = True

# model
model = models.wide_resnet50_2(pretrained=True)
model.fc = nn.Linear(2048, args.num_classes)
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
val_dataset = inat2018_loader.INAT(args.data_root, args.test_file, args.cat_file, is_train=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

# validate
criterion = nn.CrossEntropyLoss().cuda()
_, preds, im_ids = utils.validate(args, val_loader, model, criterion, True)
# write predictions to file
with open(args.op_file_name, 'w') as opfile:
    opfile.write('id,predicted\n')
    for ii in range(len(im_ids)):
        opfile.write(str(im_ids[ii]) + ',' + ' '.join(str(x) for x in preds[ii, :]) + '\n')




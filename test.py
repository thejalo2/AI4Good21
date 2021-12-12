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

from utils import AverageMeter, accuracy, save_checkpoint, train_epoch, validate
from params import Params
import data
from models import SharedEmbedderModel
import pickle

args = Params()
cudnn.benchmark = True

# model
model = SharedEmbedderModel(num_classes=8142, hidden_size=768, share_embedder=args.share_embedder).cuda()
model.inference_alpha = args.inference_alpha

# data
try:
    config = resolve_data_config({}, model=model.embedder)
except:
    config = resolve_data_config({}, model=model.embedder_rb)
val_dataset = data.INAT(args.data_root, args.test_file, args.cat_file, config, args.beta, is_train=False)
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
prec1_overall, prec3_overall, preds, im_ids = validate(args, val_loader, model, criterion, True)
# write predictions to file
with open(args.op_file_name, 'w') as opfile:
    opfile.write('id,predicted\n')
    for ii in range(len(im_ids)):
        opfile.write(str(im_ids[ii]) + ',' + ' '.join(str(x) for x in preds[ii, :]) + '\n')
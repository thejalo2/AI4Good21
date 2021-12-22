# https://github.com/samiraabnar/attention_flow
# https://github.com/google-research/vision_transformer/issues/27
# https://github.com/google-research/vision_transformer/issues/18
# https://github.com/faustomorales/vit-keras/blob/65724adcfd3979067ce24734f08df0afa745637d/vit_keras/visualize.py#L7-L45
# https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
# https://github.com/rwightman/pytorch-image-models/issues/292
# https://gist.github.com/zlapp/40126608b01a5732412da38277db9ff5

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import transforms
import cv2
import os
import os
import sys
import timm
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
from utils import AverageMeter, accuracy, save_checkpoint, train_epoch, validate
from params import Params
import data
from models import SharedEmbedderModel
import pickle
import json
import imageio


activation = {}


def get_attn_softmax(name):
    def hook(model, input, output):
        with torch.no_grad():
            input = input[0]
            B, N, C = input.shape
            qkv = (
                model.qkv(input)
                    .detach()
                    .reshape(B, N, 3, model.num_heads, C // model.num_heads)
                    .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)
            attn = (q @ k.transpose(-2, -1)) * model.scale
            attn = attn.softmax(dim=-1)
            activation[name] = attn

    return hook


# expects timm vis transformer model
def add_attn_vis_hook(model):
    for idx, module in enumerate(list(model.blocks.children())):
        module.attn.register_forward_hook(get_attn_softmax(f"attn{idx}"))


def get_mask(im, att_mat):
    # Average the attention weights across all heads.
    # att_mat,_ = torch.max(att_mat, dim=1)
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    result = (mask * im).astype("uint8")
    return result, joint_attentions, grid_size


def show_attention_map(model, model_full, img_path, shape, config):
    add_attn_vis_hook(model)
    im = Image.open(os.path.expandvars(img_path))
    im = im.resize((shape, shape))

    mu_data = config['mean']
    std_data = config['std']
    im_size = config['input_size'][1:]
    norm_aug = transforms.Normalize(mean=mu_data, std=std_data)
    tensor_aug = transforms.ToTensor()
    center_crop = transforms.CenterCrop((im_size[0], im_size[1]))
    transform = transforms.Compose([
        center_crop,
        tensor_aug,
        norm_aug
    ])

    tensor_im = transform(im).unsqueeze(0)

    print('Pred:')
    print(torch.argmax(model(tensor_im)[0]))
    # print(torch.argmax(model_full(tensor_im, None)[0]))

    logits = model(tensor_im)
    print(tensor_im.shape)

    attn_weights_list = list(activation.values())

    result, joint_attentions, grid_size = get_mask(im, torch.cat(attn_weights_list))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)
    imageio.imwrite(img_path.replace('.jpg', 'att.jpg'), result)

    probs = torch.nn.Softmax(dim=-1)(logits)
    top5 = torch.argsort(probs, dim=-1, descending=True)
    print("Prediction Label and Attention Map!\n")
    for idx in top5[0, :5]:
        print(f'{probs[0, idx.item()]:.5f} : {idx.item()}', end='')

    for i, v in enumerate(joint_attentions):
        # Attention from the output token to the input space.
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
        result = (mask * im).astype("uint8")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map_%d Layer' % (i + 1))
        _ = ax1.imshow(im)
        _ = ax2.imshow(result)

    plt.show()



model_name = 'vit_base_patch16_224'
shape = eval(model_name[-3:])

args = Params()
cudnn.benchmark = True

# model
model = SharedEmbedderModel(num_classes=8142, hidden_size=768, share_embedder=args.share_embedder)
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

img_path = r'H:\janik\Dropbox\1 - ETHZ\HS 21\5 - AI4Good\Project\bbn\attention_map_examples\3ec6df6edf697fe0164ce295d274a991.jpg'
# show_attention_map(model.embedder, model, img_path, shape, config)
m = model.embedder
m.head = model.rb
show_attention_map(m, model, img_path, shape, config)
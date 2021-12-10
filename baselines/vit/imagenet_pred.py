import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from PIL import Image
import urllib

# create model
model = timm.create_model('vit_base_patch16_224', pretrained=True)
# model = timm.create_model('vit_large_patch32_384', pretrained=True)
model.eval()

# load data transformation
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# load image
filename = r'C:\Users\janik\inat_data\train_val2018\Animalia\2176\4eb35c8fa706795d86470166d0ffa7a0.jpg'
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

# make prediction
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)

# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 3)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())

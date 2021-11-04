# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor, ToPILImage

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import json


class INaturalistDataset(object):
    def __init__(self, root, transforms, train):
        self.root = root
        self.transforms = transforms

        # load file_path, bboxes information
        if train:
            with open('train_2017_new_bboxes.json', 'r') as file:
                self.bbox_info = json.load(file)
        else:
            with open('val_2017_new_bboxes.json', 'r') as file:
                self.bbox_info = json.load(file)
        
        # only for Laptop
        self.bbox_info = self.bbox_info[:2000]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.bbox_info[idx]['file_name']
        if not os.path.isfile(img_path):
            idx = 0
            img_path = self.bbox_info[idx]['file_name']

        img = Image.open(img_path).convert("RGB")

        boxes = []
        lt_x, lt_y, width, height = self.bbox_info[idx]['bbox']
        bbox = [lt_x, lt_y, lt_x+width, lt_y+height]
        boxes.append(bbox)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels # All classes are 1 except background (= 0)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.bbox_info)


def get_model(num_classes):
    # load a object detection model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = INaturalistDataset('train_val_images', get_transform(train=True), train=True)

    # define training data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
 
    # get the model using our helper function
    model = get_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # let's train it for 10 epochs
    num_epochs = 1

    for epoch in range(int(num_epochs)):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

    print("That's it!")
    print("#### Saving the model ####")

    torch.save(model, f'{os.getcwd()}/model/trainedModel.pth')
    print("#### Model saved! Now execute tv-evaluation-code.py to run it! ####")

if __name__ == "__main__":
    main()
    # print(os.getcwd())

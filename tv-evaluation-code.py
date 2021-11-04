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


def convert_tensor_to_RGB(network_output):
   
    converted_tensor = torch.squeeze(network_output)

    return converted_tensor

def main():
    
    # use our dataset and defined transformations
    dataset_test = INaturalistDataset('train_val_images', get_transform(train=False), train=False)

    # define data loader
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    #This is the same path you stored your model
    path = f"{os.getcwd()}/model/trainedModel.pth"
    model = torch.load(path)
    model.eval()

    print("###### Running the model ######")
    model.eval()
    model.cuda()
    image = next(iter(data_loader_test))

    #Here we create a list, because the model expects a list of Tensors
    lista = []
    #It is important to send the image to CUDA, otherwise it will try to execute in the CPU
    x = image[0][0].cuda()
    lista.append(x)
    output = model(lista)

    print("### Converting output to RGB ###")
    output = convert_tensor_to_RGB(output[0].get('masks'))

    #Here, we pass the output to CPU in order to properly save the image
    output_cpu = output.cpu()

    #Just a number to order your images
    number = 2

    #Saving the images
    ToPILImage()(output_cpu).save('images/test_'+str(number)+'.png', mode='png')
    print("#### All Done! :) ####")

if __name__ == "__main__":
    main()

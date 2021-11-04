import os
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import copy

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor, ToPILImage

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import json
from tqdm import tqdm

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
        self.bbox_info = self.bbox_info[:200]


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

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-top coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-bottom coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    iymin = max(pred_box[1], gt_box[1])
    ixmax = min(pred_box[2], gt_box[2])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin, 0.)
    ih = np.maximum(iymax-iymin, 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]) * (pred_box[3]-pred_box[1]) +
           (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


### ADD Evaluation
def main():

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset_train = INaturalistDataset('train_val_images', get_transform(train=True), train=True)
    dataset_test = INaturalistDataset('train_val_images', get_transform(train=False), train=False)

    # define training data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
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
    num_epochs = 100
    save_interval = 10
    print(len(data_loader_train), len(data_loader_test))

    for epoch in range(num_epochs):
        # Train
        model.train()

        for images, targets in tqdm(data_loader_train):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # update the learning rate
        lr_scheduler.step()

        # Save the model and evaluate based on test images
        if epoch % save_interval == 0:
            print("#### Saving the model ####")
            torch.save(model, f'{os.getcwd()}/model/trainedModel_{epoch}.pth')

            model.eval()
            model.cuda()
            for image, gt in tqdm(data_loader_test):
                #Here we create a list, because the model expects a list of Tensors
                lista = []
                #It is important to send the image to CUDA, otherwise it will try to execute in the CPU
                x = image[0].cuda()
                lista.append(x)
                output = model(lista)

                # print("### Drawing Bounding Boxes ###")
                bboxes = output[0].get('boxes')

                # Pick only one bbox with highest score
                if len(output[0]['scores']) >= 1:
                  ind = torch.argmax(output[0]['scores'])
                  bbox = bboxes[ind]
                else:
                  bbox = torch.tensor([0,0,0,0])
                  
                # Draw Prediction bbox 
                pred_bbox = bbox.cpu().detach().numpy().astype(int)

                # Draw Ground-Truth bbox
                gt_bbox = gt[0]['boxes'][0].numpy().astype(int)
                total_iou += get_iou(pred_bbox, gt_bbox)

            total_iou = total_iou / len(data_loader_test)
            print(f"Epoch: {epoch}, Total_IoU: {total_iou}, Loss: {losses}")



    print("That's it!")
    print("#### Saving the Final model ####")

    torch.save(model, f'{os.getcwd()}/model/trainedModel_{epoch}.pth')
    print("#### Finished! ####")

if __name__ == "__main__":
    main()
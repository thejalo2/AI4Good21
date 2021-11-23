import argparse
import os
import json
from progressbar import progressbar
from statistics import mean,stdev
import numpy as np

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

def yolov5_to_encompassingB (Yv5entries,img_width,img_height):
    """
    Yv5entries : list of entries of the form [id , x_center, y_center, b_width, b_height] with coordinates express in ratio to the image dimensions
    img_width  : width of the image in pixels
    img_height : heigth of the image in pixels
    return     : coordinates [left, top, right, bottom] of the boxes who encompass all the boxes
    """
    boxes_left, boxes_top, boxes_right, boxes_bottom = [],[],[],[]

    for entry in Yv5entries:
        b_x_center = int(entry[1]*img_width)
        b_y_center = int(entry[2]*img_width)
        b_hwidth    = int(entry[3]*img_width/2)
        b_hheight   = int(entry[4]*img_height/2)

        boxes_left.append(   b_x_center - b_hwidth)
        boxes_top.append(    b_y_center - b_hheight)
        boxes_right.append(  b_x_center + b_hwidth)
        boxes_bottom.append( b_y_center + b_hheight)
    
    eb_left = min(boxes_left)
    eb_top  = min(boxes_top)
    eb_right = max(boxes_right)
    eb_bottom = max(boxes_bottom)

    return [eb_left, eb_top, eb_right, eb_bottom]

#########################################  MAIN  #########################################3

#parse arguments
parser = argparse.ArgumentParser(description='Evaluate average iou  ')
parser.add_argument('-g','--gt_boxes', help='path to the folder containing the txt files with the ground truth boxes in yolov5 format')
parser.add_argument('-d','--detect_boxes', help='path to the folder containing the txt files with the detected boxes in yolov5 format')
parser.add_argument('-o','--output_path', help='path to the output file containing the result of the evaluation')

args = parser.parse_args()
gt_boxes_path = args.gt_boxes
detect_boxes_path = args.detect_boxes
output_file = f"{args.output_path}.txt"

#parse files in detect_boxes  and evaluate iou: SUPPOSE THAT THE GROUND BOXES FILES AND DETECT FILES HAVE THE SAME NAME

iou = []
specie_iou = {"Plantae" : [], "Insecta" : [], "Aves" : [], "Reptilia" : [], "Mammalia" : [], "Fungi" : [], "Amphibia" : [], "Mollusca" : [], "Animalia" : [], "Arachnida" : [], "Actinopterygii" : [], "Chromista" : [], "Protozoa" : [] }


val_json_path = "/cluster/scratch/dgeissbue/INaturalist2017/val_2017_new_bboxes.json"
with open(val_json_path) as f:
    val_json = json.load(f)
image_specie_dict = {}
for entry in progressbar(val_json, redirect_stdout=True):
    moto = entry["file_name"].split("/")
    entry_specie = moto[1]
    entry_imageName = moto[-1].split(".")[0]
    image_specie_dict[entry_imageName] = entry_specie



dummy_img_width  = 640
dummy_img_height = 640
detect_txt_files = os.listdir(detect_boxes_path)
gt_txt_files = os.listdir(gt_boxes_path)

for file_name in progressbar(gt_txt_files, redirect_stdout=True):
    if file_name.split(".")[-1] == "txt": #it is a txt files
        
        Yv5_detect_entries = []
        detect_file = os.path.join(detect_boxes_path,file_name)

        if os.path.exists(detect_file): #if the detect file exist in detect folder parse it
            with open(os.path.join(detect_boxes_path,file_name),'r') as detect_f:
                for line in detect_f:

                    Yv5_detect_entries.append([float(element) for element in line.split(" ")])

        else:  #otherwise return the box encompassing the whole image
            Yv5_detect_entries.append([0, 0.5, 0.5, 1, 1])


        #parse file in ground thruth folder:
        Yv5_gt_entries = []
        with open(os.path.join(gt_boxes_path,file_name),'r') as gt_f:
            for line in gt_f:

                Yv5_gt_entries.append([float(element) for element in line.split(" ")])

        pred_box = yolov5_to_encompassingB(Yv5_detect_entries, dummy_img_width, dummy_img_height)
        gt_box   = yolov5_to_encompassingB(Yv5_gt_entries, dummy_img_width, dummy_img_height)
        value = get_iou(pred_box,gt_box)
        iou.append(value)

        image_name = file_name.split(".")[0]
        specie_iou[image_specie_dict[image_name]].append(value)


with open(output_file,'w') as output_f:
    entry = f"***iou statistics***\n avg : {mean(iou)}\n minimum : {min(iou)}\n maximum : {max(iou)}\n standart deviation : {stdev(iou)}\n"
    for specie in specie_iou:
        value = 0 
        if len(specie_iou[specie]) > 0 :
            value = mean(specie_iou[specie])
        print(f"{specie} : {value}")
    output_f.write(entry)

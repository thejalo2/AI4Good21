import argparse
import os
import shutil
import json
from progressbar import progressbar

#had to impose a limit on the number of files due to disk quota on Euler cluster
maxTrainingFiles = 400000
maxValidationFiles = 50000

def Json2Yolovtxt(Json_entry,coco_path,dataset_type):
    """
    Create a txt file in in the Yolov5 format 
    Json_entry : dict of the form :
    {
        "file_name" : path to the image from the folder containing train_val_images
        "width"   :   width of the image in pixels
        "height"  :   height of the image in pixels
        "id"      :   identification number
        "bbox"  :     4 tuples (left,top,right,bottom) containing the coordinates of the bounding boxe in pixels
    }
    coco_path : path to the folder containing coco data
    dataset_type : whether the label is for training (train) or validation (val)
    """
    #here the coordinates are expressed in pixels
    b_x      = Json_entry["bbox"][0] #x of left-top 
    b_y      = Json_entry["bbox"][1] #y of left-top
    b_width  = Json_entry["bbox"][2] #width of the box
    b_height = Json_entry["bbox"][3] #height of the box
    img_width  =  Json_entry["width"]
    img_height =  Json_entry["height"]

    #converting them into ratio over image dimension
    B_x      = int(b_x + (b_width/2))/img_width
    B_y      = int(b_y + (b_height/2))/img_height
    B_width  = b_width/img_width
    B_height = b_height/img_height

    label_id = 0

    yolov_entry = f"{label_id} {B_x} {B_y} {B_width} {B_height}"
    image_name = Json_entry["file_name"].split("/")[-1]
    image_id = image_name.split('.')[0]
    label_file_rel_path = f"labels/{dataset_type}/{image_id}.txt"
    label_file_path = os.path.join(coco_path,label_file_rel_path)

    with open(label_file_path,"w") as label_file:
        label_file.write(yolov_entry)

def savingImage(Json_entry,coco_path,image_set_path,dataset_type):
    """
    Search the image and save it in the correct folder
    Json_entry : dict of the form :
    {
        "file_name" : path to the image from the folder containing train_val_images
        "width"   :   width of the image in pixels
        "height"  :   height of the image in pixels
        "id"      :   identification number
        "bbox"  :     4 tuples (left,top,right,bottom) containing the coordinates of the bounding boxe in pixels
    }
    coco_path : path to the folder containing coco data
    dataset_type : whether the label is for training (train) or validation (val)
    """
    file_path = Json_entry["file_name"]
    src_file = os.path.join(image_set_path,file_path)
    dest_dir = os.path.join(coco_path,f"images/{dataset_type}/")

    shutil.copy(src_file,dest_dir)

parser = argparse.ArgumentParser(description='Evaluate average iou  ')
parser.add_argument('-t','--training_json', help='path to json file containing training data')
parser.add_argument('-v','--validation_json', help='path to json file containing validation data')
parser.add_argument('-d','--directory_data', help='path to directory containing the folder train_val_images')
parser.add_argument('-o','--output_name', help='name of the output folder')

#parse arguments
args = parser.parse_args()
train_json_path = args.training_json
val_json_path = args.validation_json
data_directory = args.directory_data
output_folder = args.output_name

#create coco folder:
os.mkdir(output_folder)
os.mkdir(os.path.join(output_folder,"images"))
os.mkdir(os.path.join(output_folder,"images/train"))
os.mkdir(os.path.join(output_folder,"images/val"))
os.mkdir(os.path.join(output_folder,"labels"))
os.mkdir(os.path.join(output_folder,"labels/train"))
os.mkdir(os.path.join(output_folder,"labels/val"))

#go through training set
print("opening training json...")
with open(train_json_path) as f:
    train_json = json.load(f)
    N_train_files = 0
    print("training set : creating txt files and saving images ...")
    for entry in progressbar(train_json, redirect_stdout=True):
        Json2Yolovtxt(entry,output_folder,'train')
        savingImage(entry,output_folder,data_directory,'train')
        N_train_files += 1
        if N_train_files > maxTrainingFiles:
            break 
#go through validation set
print("opening validation json...")
with open(val_json_path) as f:
    val_json = json.load(f)
    N_val_files = 0
    print("validation set : creating txt files and saving images ...")
    for entry in progressbar(val_json, redirect_stdout=True):
        Json2Yolovtxt(entry,output_folder,'val')
        savingImage(entry,output_folder,data_directory,'val')
        N_val_files += 1
        if N_val_files > maxValidationFiles:
            break

import argparse
import os
import shutil
import json
from progressbar import progressbar

parser = argparse.ArgumentParser(description='Evaluate average iou  ')
parser.add_argument('-t','--training_json', help='path to json file containing training data')
parser.add_argument('-v','--validation_json', help='path to json file containing validation data')
parser.add_argument('-d','--directory_data', help='path to directory containing the folder train_val_images')


#parse arguments
args = parser.parse_args()
train_json_path = args.training_json
val_json_path = args.validation_json
data_directory = args.directory_data



#create a list of training file name :
print("load train json")
boxed_images = []
with open(train_json_path) as f :
    J = json.load(f)
    for entry in J:
        file_name = entry["file_name"].split("/")[-1]
        boxed_images.append(file_name)

#create a list of validation file name :
print("load val json")
boxed_val_images = []
with open(val_json_path) as f :
    J = json.load(f)
    for entry in J:
        file_name = entry["file_name"].split("/")[-1]
        boxed_images.append(file_name)


print("go through train_val_dataset")
images_boxed = 0
images_unboxed = 0
total_images = 675170
current_count = 0
for root, dirs, files in os.walk(data_directory, topdown=True):
    for file_name in files:  
        if file_name not in boxed_images and file_name.split(".")[-1] == "jpg":
            file_path = os.path.join(root,file_name)
            images_unboxed += 1
            current_count += 1
            os.remove(file_path)
        elif file_name.split(".")[-1] == "jpg":
            images_boxed +=1
            current_count += 1
        
        if current_count%1000 == 0:
            print(f"{current_count}/{total_images}")

print("------------------------------")
print(f"#boxed : {images_boxed}")
print(f"#unboxed : {images_unboxed}")
print(f"total : {images_unboxed+images_boxed}")
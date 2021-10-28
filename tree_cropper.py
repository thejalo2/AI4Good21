"""
Go through a folder and crop the images inside, while copying the folder and replacing the imags 
by their cropped versions
"""
import argparse
import os
import tqdm

parser = argparse.ArgumentParser(description='Go through a folder and crop the images inside, while copying the folder and replacing the imags by their cropped versions')
parser.add_argument('-s','--source_path', help='path to the folder containing images to be cropped')
parser.add_argument('-y','--yolov5_path', help='path to the folder containing Yolov5')
args = parser.parse_args()

source_path = args.source_path
yolov5_path = args.yolov5_path

#create output file
output_path = os.path.join(os.getcwd(),"images_cropped")
os.mkdir(output_path)

#go through source folder
for subFolderRoot, foldersWithinSubFolder, files in tqdm.tqdm(os.walk(source_path, topdown=True)):

        #for all the files in the folder we crop them
        if len(files)>0:
            #print(f"python \"{yolov5_path}/yolov5/detect.py\" --source \"{subFolderRoot}\" --weights yolov5x.pt --conf 0.2 --save-txt")
            os.system(f"python \"{yolov5_path}/yolov5/detect.py\" --source \"{subFolderRoot}\" --weights yolov5x.pt --conf 0.2 --save-txt")
            boxes_information_path = f"{yolov5_path}/yolov5/runs/detect/exp/labels"
            rel_path = os.path.relpath(subFolderRoot, source_path)
            if rel_path == ".":
                rel_path = ""
            output_folder = os.path.join(output_path,rel_path)
            #print(f"python cropper.py -s \"{subFolderRoot}\" -b \"{boxes_information_path}\" -o \"{output_folder}\"")
            os.system(f"python cropper.py -s \"{subFolderRoot}\" -b \"{boxes_information_path}\" -o \"{output_folder}\"")
            os.system(f"rm -rf \"{yolov5_path}/yolov5/runs/detect/exp\"")
            
        #for the folders we copy them in the output with the same structure
        for folderNameWithinSubFolder in foldersWithinSubFolder:
            rel_path = os.path.relpath(subFolderRoot, source_path)
            if rel_path == ".":
                rel_path = ""
            folder_rel_path = os.path.join(rel_path,folderNameWithinSubFolder)
            new_folder = os.path.join(output_path,folder_rel_path)
            os.mkdir(new_folder)
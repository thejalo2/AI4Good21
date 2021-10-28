"""
Retrieve the bounding box from output of Yolov5 and crop the initial images 
so with the smallest bounding box encompassing all detection boxes
"""
import argparse
import os,io
from PIL import Image 
import glob


parser = argparse.ArgumentParser(description='Retrieve the bounding box from output of Yolov5 and crop the initial images so with the smallest bounding box encompassing all detection boxes ')
parser.add_argument('-s','--source_path', help='path to the folder containing images to be cropped')
parser.add_argument('-o','--output_path', help='path to the folder where cropped image are saved')
parser.add_argument('-b','--boxes_path',  help='path to the folder containing box dimensions')
args = parser.parse_args()

#print(f"SOURCE_IMAGES_PATH = {args.source_path}")
#print(f"BOUNDING_BOXES_PATH = {args.boxes_path}")
#print(f"OUTPUT_PATH = {args.output_path}")


files = os.listdir(args.source_path)

for fileName in files:
    if  fileName.endswith(".png") or  fileName.endswith(".jpeg") or  fileName.endswith(".jpg") :
        #generate the txt file name for the box dimension files
        name = os.path.splitext(fileName)[0]
        box_file =  name + ".txt"

        #print(fileName)
        img = Image.open(args.source_path+"/"+fileName)
        width, height = img.size
        bounds_left = []
        bounds_top = []
        bounds_right = []
        bounds_bottom = []

        with io.open(args.boxes_path+"/"+box_file, mode="r", encoding="utf-8") as f:
            for line in f:
                splitted_line = line.split()
                center_box_x = int(float(splitted_line[1])*width)
                center_box_y = int(float(splitted_line[2])*height)
                half_width_box = int(float(splitted_line[3])*width/2)
                half_height_box = int(float(splitted_line[4])*height/2)

                left = center_box_x-half_width_box #position x of the top-left corner of box (from the left)
                top =  center_box_y -half_height_box #position y of the top-left corner of the box(from the top)
                right = center_box_x + half_width_box #position x of the bottom-right corner of box (from the left)
                bottom = center_box_y + half_height_box #position y of the bottom-right corner of the box(from the top)
            
                bounds_left.append(left)
                bounds_top.append(top)
                bounds_right.append(right)
                bounds_bottom.append(bottom)

        left = min(bounds_left)
        top = min(bounds_top)
        right = max(bounds_right)
        bottom = max(bounds_bottom)
        img_res = img.crop((left,top,right,bottom))
        img_res.save(args.output_path+"/"+name+".png")

        




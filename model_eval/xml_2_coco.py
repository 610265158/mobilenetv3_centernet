#-*-coding:utf-8-*-
####transform from xml to json

import os
import xml.etree.cElementTree as et
import json
import argparse
import shutil
import traceback
import random
import numpy as np
import cv2

def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList


# load train/val split used in the training
annotation_path = './val.txt'


with open(annotation_path) as f:
    lines = f.readlines()

# initialize the json data for the dataset
data = {}
cls_person = 0

test_data = {}
test_data['licenses'] = []
test_data['info'] = []
test_data['categories'] = [{'id': cls_person, 'name': 'person', 'supercategory': 'person'}]
test_data['images'] = []
test_data['annotations'] = []

# process xml files
counter=1
anno_id = 0
img_id = 0
for line in lines:
    counter+=1
    if counter%1000==0:
        print('%d/%d images processed'%(counter, len(lines)))
    try:

        file_str,label = line.rstrip().rsplit('| ')

        labels = label.split(' ')
        boxes = []

        for label in labels:
            
            bbox = np.array(label.split(','), dtype=np.float)
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])


        #file_name = root.find('filename').text
        file_name = file_str
        image_id = img_id

        img=cv2.imread(file_name)
        img_height,img_width,_=img.shape


        img_entry = {'file_name': file_name, 'id': image_id, 'height': img_height, 'width': img_width}
        test_data['images'].append(img_entry)

        img_id += 1

        for box in boxes:

            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])

            anno_entry = {'image_id': image_id, 'category_id': cls_person, 'id': anno_id,\
                        'iscrowd': 0, 'area': int(xmax-xmin) * int(ymax-ymin),\
                        'bbox': [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]}
            test_data['annotations'].append(anno_entry)

            anno_id += 1
    except Exception as ex:
        msg = "err:%s" % ex
        print(msg)
        traceback.print_exc()


with open('./model_eval/DatasetTest_cocoStyle.json', 'w') as outfile:
    json.dump(test_data, outfile)
import numpy as np
import os

from lib.dataset.coco_data import BoxInfo

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mscocodir', type=str,default='../pubdata/mscoco', help='detect with coco or face',required=False)
args = parser.parse_args()

coco_dir=args.mscocodir

train_im_path = os.path.join(coco_dir,'train2017')
train_ann_path =  os.path.join(coco_dir,'annotations/instances_train2017.json')
val_im_path =  os.path.join(coco_dir,'val2017')
val_ann_path =  os.path.join(coco_dir,'annotations/instances_val2017.json')



train_data=BoxInfo(train_im_path,train_ann_path)


fw = open('train.txt', 'w')
for meta in train_data.metas:
    fname, boxes = meta.img_url, meta.bbox



    tmp_str = ''
    tmp_str =tmp_str+ fname+'|'

    for box in boxes:
        data = ' %d,%d,%d,%d,%d'%(box[0], box[1], box[2],  box[3],box[4])
        tmp_str=tmp_str+data
    if len(boxes) == 0:
        print(tmp_str)
        continue
    ####err box?
    if box[2] <= 0 or box[3] <= 0:
        pass
    else:
        fw.write(tmp_str + '\n')
fw.close()






val_data=BoxInfo(val_im_path,val_ann_path)

fw = open('val.txt', 'w')
for meta in val_data.metas:
    fname, boxes = meta.img_url, meta.bbox

    tmp_str = ''
    tmp_str = tmp_str + fname + '|'

    for box in boxes:
        data = ' %d,%d,%d,%d,%d' % (box[0], box[1], box[2], box[3], box[4])
        tmp_str = tmp_str + data
    if len(boxes) == 0:
        print(tmp_str)
        continue
    ####err box?
    if box[2] <= 0 or box[3] <= 0:
        pass
    else:
        fw.write(tmp_str + '\n')
fw.close()

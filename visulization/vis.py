import sys
sys.path.append('.')

import cv2
import os
import time


from lib.core.api.face_detector import FaceDetector
from train_config import config as cfg

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--style', type=str,default='coco', help='detect with coco or face',required=False)
parser.add_argument('--imgDir', type=str,default='../pubdata/mscoco/val2017', help='the image dir to detect')
parser.add_argument('--thres', type=float,default=0.3, help='the thres for detect')
args = parser.parse_args()

data_dir=args.imgDir
style=args.style
thres=args.thres

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
detector = FaceDetector(['./model/detector.pb'])
coco_map = {0: (1, 'person'), 1: (2, 'bicycle'), 2: (3, 'car'), 3: (4, 'motorcycle'), 4: (5, 'airplane'), 5: (6, 'bus'),
            6: (7, 'train'), 7: (8, 'truck'), 8: (9, 'boat'), 9: (10, 'traffic shufflenet'), 10: (11, 'fire hydrant'),
            11: (13, 'stop sign'), 12: (14, 'parking meter'), 13: (15, 'bench'), 14: (16, 'bird'), 15: (17, 'cat'),
            16: (18, 'dog'), 17: (19, 'horse'), 18: (20, 'sheep'), 19: (21, 'cow'), 20: (22, 'elephant'),
            21: (23, 'bear'), 22: (24, 'zebra'), 23: (25, 'giraffe'), 24: (27, 'backpack'), 25: (28, 'umbrella'),
            26: (31, 'handbag'), 27: (32, 'tie'), 28: (33, 'suitcase'), 29: (34, 'frisbee'), 30: (35, 'skis'),
            31: (36, 'snowboard'), 32: (37, 'sports ball'), 33: (38, 'kite'), 34: (39, 'baseball bat'),
            35: (40, 'baseball glove'),
            36: (41, 'skateboard'), 37: (42, 'surfboard'), 38: (43, 'tennis racket'), 39: (44, 'bottle'),
            40: (46, 'wine glass'),
            41: (47, 'cup'), 42: (48, 'fork'), 43: (49, 'knife'), 44: (50, 'spoon'), 45: (51, 'bowl'),
            46: (52, 'banana'), 47: (53, 'apple'), 48: (54, 'sandwich'), 49: (55, 'orange'), 50: (56, 'broccoli'),
            51: (57, 'carrot'), 52: (58, 'hot dog'), 53: (59, 'pizza'), 54: (60, 'donut'), 55: (61, 'cake'),
            56: (62, 'chair'), 57: (63, 'couch'), 58: (64, 'potted plant'), 59: (65, 'bed'), 60: (67, 'dining table'),
            61: (70, 'toilet'), 62: (72, 'tv'), 63: (73, 'laptop'), 64: (74, 'mouse'), 65: (75, 'remote'),
            66: (76, 'keyboard'), 67: (77, 'cell phone'), 68: (78, 'microwave'), 69: (79, 'oven'), 70: (80, 'toaster'),
            71: (81, 'sink'), 72: (82, 'refrigerator'), 73: (84, 'book'), 74: (85, 'clock'), 75: (86, 'vase'),
            76: (87, 'scissors'), 77: (88, 'teddy bear'), 78: (89, 'hair drier'), 79: (90, 'toothbrush')}

def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList


def cocodetect(data_dir):
    success_cnt=0
    count = 0

    pics = []
    GetFileList(data_dir,pics)

    pics = [x for x in pics if 'jpg' in x or 'png' in x or 'jpeg' in x]
    #pics.sort()

    for pic in pics:
        print(pic)
        try:
            img=cv2.imread(pic)
            #cv2.imwrite('tmp.png',img)
            img_show = img.copy()
        except:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        star=time.time()
        boxes=detector(img,thres,input_shape=(cfg.DATA.hin,cfg.DATA.win))

        print(boxes.shape[0])
        if boxes.shape[0]==0:
            print(pic)

        for box_index in range(boxes.shape[0]):

            bbox = boxes[box_index]

            cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)
            str_draw = '%s:%.2f' %(coco_map[int(bbox[5])][1],bbox[4])
            cv2.putText(img_show, str_draw, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 0, 255), 2)


        cv2.namedWindow('res',0)
        cv2.imshow('res',img_show)
        cv2.waitKey(0)

    print(success_cnt,'decoded')
    print(count)


def camdetect():
    cap = cv2.VideoCapture(0)

    while True:

        ret, img = cap.read()
        img_show = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        star=time.time()
        boxes=detector(img,0.5,input_shape=(640,640))


        print(boxes.shape[0])


        for box_index in range(boxes.shape[0]):

            bbox = boxes[box_index]

            cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 0, 0), 8)
            # cv2.putText(img_show, str(bbox[4]), (int(bbox[0]), int(bbox[1]) + 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (255, 0, 255), 2)
            #
            # cv2.putText(img_show, str(int(bbox[5])), (int(bbox[0]), int(bbox[1]) + 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 0, 255), 2)


        cv2.namedWindow('res',0)
        cv2.imshow('res',img_show)
        cv2.waitKey(0)
    print(count)

def facedetect(data_dir):
    success_cnt=0
    count = 0

    pics = []
    GetFileList(data_dir,pics)

    pics = [x for x in pics if 'jpg' in x or 'png' in x or 'jpeg' in x]
    #pics.sort()

    for pic in pics:
        print(pic)
        try:
            img=cv2.imread(pic)
            #cv2.imwrite('tmp.png',img)
            img_show = img.copy()
        except:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        star=time.time()
        boxes=detector(img,thres,input_shape=(cfg.DATA.hin,cfg.DATA.win))

        print(boxes.shape[0])
        if boxes.shape[0]==0:
            print(pic)

        for box_index in range(boxes.shape[0]):

            bbox = boxes[box_index]

            cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)
            str_draw = '%s:%.2f' %(coco_map[int(bbox[5])][1],bbox[4])
            cv2.putText(img_show, str_draw, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 0, 255), 2)


        cv2.namedWindow('res',0)
        cv2.imshow('res',img_show)
        cv2.waitKey(0)

    print(success_cnt,'decoded')
    print(count)
if __name__=='__main__':

    if style=='coco':
        cocodetect(data_dir)
    else:
        facedetect(data_dir)

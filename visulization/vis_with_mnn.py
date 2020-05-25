# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
import sys
sys.path.append('.')

import numpy as np
import MNN
import cv2
import os

from visulization.coco_id_map import coco_map
from train_config import config as cfg

def preprocess( image, target_height, target_width, label=None):
    ###sometimes use in objs detects
    h, w, c = image.shape

    bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype)

    scale_y = target_height / h
    scale_x = target_width / w

    scale = min(scale_x, scale_y)

    image = cv2.resize(image, None, fx=scale, fy=scale)

    h_, w_, _ = image.shape

    dx = (target_width - w_) // 2
    dy = (target_height - h_) // 2
    bimage[dy:h_ + dy, dx:w_ + dx, :] = image

    return bimage, scale, scale, dx, dy



def inference(mnn_model_path,img_dir,thres=0.3):
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter(mnn_model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    img_list=os.listdir(img_dir)
    for pic in img_list:
        image = cv2.imread(os.path.join(img_dir,pic))
        #cv2 read as bgr format
        image = image[..., ::-1]
        #change to rgb format

        image,_,_,_,_ = preprocess(image,target_height=cfg.DATA.hin,target_width=cfg.DATA.win)
        image_show=image.copy()

        image = image.astype(np.float32)

        tmp_input = MNN.Tensor((1, cfg.DATA.hin, cfg.DATA.win,3 ), MNN.Halide_Type_Float,\
                        image, MNN.Tensor_DimensionType_Tensorflow)
        #construct tensor from np.ndarray
        input_tensor.copyFrom(tmp_input)

        ### caution!!!!!!!!!!!!!!!! the model is nhwc

        interpreter.resizeSession(session)
        interpreter.runSession(session)

        output_tensor = interpreter.getSessionOutputAll(session)

        boxes=output_tensor['tower_0/concat_1'].getData()
        print(boxes)
        boxes=np.reshape(boxes,newshape=[100,6])
        print(boxes.shape)
        for i in range(len(boxes)):
            bbox = boxes[i]
            print(bbox)
            if bbox[4]>thres:



                cv2.rectangle(image_show, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)
                str_draw = '%s:%.2f' % (coco_map[int(bbox[5])][1], bbox[4])
                cv2.putText(image_show, str_draw, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 0, 255), 2)

        cv2.imshow('mnn result',image_show)
        cv2.waitKey(0)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mnn_model', type=str, default='./centernet.mnn', help='the mnn model ', required=False)
    parser.add_argument('--imgDir', type=str, default='../pubdata/mscoco/val2017', help='the image dir to detect')
    parser.add_argument('--thres', type=float, default=0.3, help='the thres for detect')
    args = parser.parse_args()

    data_dir = args.imgDir
    model_path=args.mnn_model
    thres=args.thres
    inference(model_path,data_dir,thres)

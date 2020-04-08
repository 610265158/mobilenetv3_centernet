#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from lib.core.anchor.box_utils import batch_decode,batch_decode_fix

from lib.core.model.net.shufflenet.backbone import shufflenetv2_ssd
from lib.core.model.net.mobilenetv3.backbone import mobilenetv3_ssd
from lib.core.model.net.mobilenet.backbone import mobilenet_ssd
from lib.core.model.net.resnet.backbone import resnet_ssd
from lib.core.model.loss.centernet_loss import loss

from train_config import config as cfg

from lib.helper.logger import logger

from lib.core.model.head.centernet_head import CenternetHead

class Centernet():

    def __init__(self,):
        if "ShufflenetV2"  in cfg.MODEL.net_structure:
            self.ssd_backbone=shufflenetv2_ssd                 ### it is a func
        elif "MobilenetV2" in cfg.MODEL.net_structure:
            self.ssd_backbone = mobilenet_ssd
        elif "MobilenetV3" in cfg.MODEL.net_structure:
            self.ssd_backbone = mobilenetv3_ssd
        elif "resnet_v2_50" in cfg.MODEL.net_structure:
            self.ssd_backbone = resnet_ssd
        self.head=CenternetHead()                         ### it is a class

    def forward(self,inputs,cls_hm,reg_hm,l2_regulation,training_flag):

        ###preprocess
        inputs=self.preprocess(inputs)

        ### extract feature maps
        origin_fms=self.ssd_backbone(inputs,training_flag)

        reg, cls = self.head(origin_fms, l2_regulation, training_flag)

        ### calculate loss
        reg_loss, cls_loss = loss(reg, cls, reg_hm,cls_hm,'focal_loss')

        ###### adjust the anchors to the image shape, but it trains with a fixed h,w

        # if not cfg.MODEL.deployee:
        #     ##adaptive anchor, more time consume
        #     h = tf.shape(inputs)[1]
        #     w = tf.shape(inputs)[2]
        #     anchors_ = get_all_anchors_fpn(max_size=[h, w])
        #     anchors_decode_=None
        # else:
        #     ###fix anchor
        #     anchors_ = anchor_tools.anchors /np.array([cfg.DATA.win,cfg.DATA.hin,cfg.DATA.win,cfg.DATA.hin])
        #     anchors_decode_ = anchor_tools.decode_anchors /np.array([cfg.DATA.win,cfg.DATA.hin,cfg.DATA.win,cfg.DATA.hin])/5.
        #
        # self.postprocess(reg, cls, anchors_, anchors_decode_)

        return reg_loss,cls_loss

    def preprocess(self,image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            image=image/255.
        return image

    def postprocess(self, reg,cls):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            where N = max_boxes.
        """

        with tf.name_scope('postprocessing'):

            cls=slim.max_pool2d(cls,kernel_size=(3,3),stride=1,padding='SAME')






#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np

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

    def postprocess(self, box_encodings, cls, anchors, anchors_decode):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            where N = max_boxes.
        """

        with tf.name_scope('postprocessing'):

            if anchors_decode is None:
                boxes = batch_decode(box_encodings, anchors)
            else:
                boxes = batch_decode_fix(box_encodings, anchors, anchors_decode)
            # if the images were padded we need to rescale predicted boxes:

            # it has shape [batch_size, num_anchors, 4]
            # scores = tf.nn.softmax(cls, axis=2)  ##ignore the bg
            scores = cls[:, :, 1:]
            scores = tf.nn.sigmoid(scores)

        if "coreml" == cfg.MODEL.deployee:
            ###this branch is for coreml

            boxes = tf.identity(boxes, name='boxes')
            scores = tf.identity(scores, name='scores')

            labels = tf.identity(scores, name='labels')  ## no use
        elif "mnn" == cfg.MODEL.deployee:

            label_with_max_score = tf.argmax(scores, axis=2)

            scores = tf.reduce_max(scores, axis=2, keep_dims=True)

            ##this branch is for mnn
            boxes = tf.squeeze(boxes, axis=[0])
            scores = tf.squeeze(scores, axis=[0, 2])
            labels = tf.squeeze(label_with_max_score, axis=[0])
            selected_indices = tf.image.non_max_suppression(
                boxes, scores, cfg.MODEL.max_box, cfg.MODEL.iou_thres, cfg.MODEL.score_thres)

            boxes = tf.gather(boxes, selected_indices)
            scores = tf.gather(scores, selected_indices)
            labels = tf.gather(labels, selected_indices)

            num_boxes = tf.cast(tf.shape(boxes)[0], dtype=tf.int32)
            zero_padding = cfg.MODEL.max_box - num_boxes

            boxes = tf.pad(boxes, [[0, zero_padding], [0, 0]])

            scores = tf.expand_dims(scores, axis=-1)
            scores = tf.pad(scores, [[0, zero_padding], [0, 0]])

            labels = tf.expand_dims(labels, axis=-1)
            labels = tf.pad(labels, [[0, zero_padding], [0, 0]])

            boxes = tf.identity(boxes, name='boxes')
            scores = tf.identity(scores, name='scores')
            labels = tf.identity(labels, name='labels')
        else:
            ###this branch is for tf

            label_with_max_score = tf.argmax(scores, axis=2)

            scores = tf.reduce_max(scores, axis=2, keep_dims=True)

            boxes = tf.identity(boxes, name='boxes')
            scores = tf.identity(scores, name='scores')
            labels = tf.identity(label_with_max_score, name='labels')

        return tf.concat([boxes, scores], axis=-1)




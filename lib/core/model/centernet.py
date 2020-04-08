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

        self.top_k_results_output=100
    def forward(self,inputs,cls_hm,reg_hm,num_gt,l2_regulation,training_flag):

        ###preprocess
        inputs=self.preprocess(inputs)

        ### extract feature maps
        origin_fms=self.ssd_backbone(inputs,training_flag)

        reg, cls = self.head(origin_fms, l2_regulation, training_flag)

        ### calculate loss
        reg_loss, cls_loss = loss(reg, cls, reg_hm,cls_hm,num_gt)

        boxes = tf.identity(cls, name='keypoints')

        #self.postprocess(reg,cls)
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

    def postprocess(self, size,keypoints):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            where N = max_boxes.
        """

        with tf.name_scope('postprocessing'):
            pshape = [tf.shape(keypoints)[1], tf.shape(keypoints)[2]]
            h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
            w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
            [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)
            keypoints = tf.sigmoid(keypoints)

            meshgrid_y = tf.expand_dims(meshgrid_y, axis=-1)
            meshgrid_x = tf.expand_dims(meshgrid_x, axis=-1)
            center = tf.concat([meshgrid_y, meshgrid_x], axis=-1)
            category = tf.expand_dims(tf.squeeze(tf.argmax(keypoints, axis=-1, output_type=tf.int32)), axis=-1)
            meshgrid_xyz = tf.concat([tf.zeros_like(category), tf.cast(center, tf.int32), category], axis=-1)
            keypoints = tf.gather_nd(keypoints, meshgrid_xyz)
            keypoints = tf.expand_dims(keypoints, axis=0)
            keypoints = tf.expand_dims(keypoints, axis=-1)
            keypoints_peak = self._max_pooling(keypoints, 3, 1)
            keypoints_mask = tf.cast(tf.equal(keypoints, keypoints_peak), tf.float32)
            keypoints = keypoints * keypoints_mask
            scores = tf.reshape(keypoints, [-1])
            class_id = tf.reshape(category, [-1])
            bbox_yx = tf.reshape(center, [-1, 2])
            bbox_hw = tf.reshape(size, [-1, 2])
            score_mask = scores > self.score_threshold
            scores = tf.boolean_mask(scores, score_mask)
            class_id = tf.boolean_mask(class_id, score_mask)
            bbox_yx = tf.boolean_mask(bbox_yx, score_mask)
            bbox_hw = tf.boolean_mask(bbox_hw, score_mask)
            bbox = tf.concat([bbox_yx-bbox_hw/2., bbox_yx+bbox_hw/2.], axis=-1) * 4
            num_select = tf.cond(tf.shape(scores)[0] > self.top_k_results_output, lambda: self.top_k_results_output, lambda: tf.shape(scores)[0])
            select_scores, select_indices = tf.nn.top_k(scores, num_select)
            select_class_id = tf.gather(class_id, select_indices)
            select_bbox = tf.gather(bbox, select_indices)

            boxes = tf.identity(select_bbox, name='boxes')
            scores = tf.identity(select_scores, name='scores')
            labels = tf.identity(select_class_id, name='labels')

            return select_scores,select_bbox,select_class_id




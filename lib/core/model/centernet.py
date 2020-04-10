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

        size, kps = self.head(origin_fms, l2_regulation, training_flag)
        kps= tf.nn.sigmoid(kps)
        ### calculate loss
        reg_loss, cls_loss = loss(size, kps, reg_hm,cls_hm,num_gt)

        kps = tf.identity(kps, name='keypoints')

        self.postprocess(size,kps)
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

        def nms(heat, kernel=3):
            hmax = tf.layers.max_pooling2d(heat, kernel, 1, padding='same')
            keep = tf.cast(tf.equal(heat, hmax), tf.float32)
            return heat * keep

        def topk(hm, K=100):
            batch, height, width, cat = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
            # [b,h*w*c]
            scores = tf.reshape(hm, (batch, -1))
            # [b,k]
            topk_scores, topk_inds = tf.nn.top_k(scores, k=K)
            # [b,k]
            topk_clses = topk_inds % cat
            topk_xs = tf.cast(topk_inds // cat % width, tf.float32)
            topk_ys = tf.cast(topk_inds // cat // width, tf.float32)
            topk_inds = tf.cast(topk_ys * tf.cast(width, tf.float32) + topk_xs, tf.int32)

            return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

        def decode(heat, wh, reg=None, K=100):
            batch, height, width, cat = tf.shape(heat)[0], tf.shape(heat)[1], tf.shape(heat)[2], tf.shape(heat)[3]
            heat = nms(heat)
            scores, inds, clses, ys, xs = topk(heat, K=K)

            if reg is not None:
                reg = tf.reshape(reg, (batch, -1, tf.shape(reg)[-1]))
                # [b,k,2]
                reg = tf.batch_gather(reg, inds)
                xs = tf.expand_dims(xs, axis=-1) + reg[..., 0:1]
                ys = tf.expand_dims(ys, axis=-1) + reg[..., 1:2]
            else:
                xs = tf.expand_dims(xs, axis=-1) + 0.5
                ys = tf.expand_dims(ys, axis=-1) + 0.5

            # [b,h*w,2]
            wh = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
            # [b,k,2]
            wh = tf.batch_gather(wh, inds)

            clses = tf.cast(tf.expand_dims(clses, axis=-1), tf.float32)
            scores = tf.expand_dims(scores, axis=-1)

            xmin = xs - wh[..., 0:1] / 2
            ymin = ys - wh[..., 1:2] / 2
            xmax = xs + wh[..., 0:1] / 2
            ymax = ys + wh[..., 1:2] / 2


            ##mul by stride 4
            bboxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)*4



            # [b,k,6]
            detections = tf.concat([bboxes, scores, clses], axis=-1)

            bboxes = tf.identity(bboxes, name='boxes')
            scores = tf.identity(scores, name='scores')
            labels = tf.identity(clses, name='labels')  ## no use
            return detections


        #with tf.name_scope('postprocessing'):

        decode(keypoints,size)







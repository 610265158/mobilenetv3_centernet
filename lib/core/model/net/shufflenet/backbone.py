#-*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.net.shufflenet.shufflenetv2 import ShufflenetV2
from lib.core.model.net.shufflenet.shufflenetv2 import shufflenet_arg_scope

from lib.core.model.fpn.seperateconv_fpn import create_fpn_net

def shufflenetv2_ssd(image,is_training=True):

    arg_scope = shufflenet_arg_scope(weight_decay=cfg.TRAIN.weight_decay_factor)

    with tf.contrib.slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            shufflenet_fms = ShufflenetV2(image,is_training=is_training)

            for fm in shufflenet_fms:
                print(fm)
            if cfg.MODEL.fpn:
                mobilebet_fms=create_fpn_net(shufflenet_fms,dims_list=cfg.MODEL.fpn_dims)

    return mobilebet_fms

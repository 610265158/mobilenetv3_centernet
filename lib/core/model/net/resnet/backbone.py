import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.net.resnet.resnet_v2 import resnet_v2_50
from lib.core.model.net.resnet.resnet_utils import resnet_arg_scope

from lib.core.model.fpn.plain_fpn import create_fpn_net

def resnet_ssd(image,is_training=True):

    arg_scope = resnet_arg_scope(weight_decay=cfg.TRAIN.weight_decay_factor)

    with tf.contrib.slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            _,endpoints = resnet_v2_50(image, is_training=is_training,global_pool=False,num_classes=None)

            for k, v in endpoints.items():
                print('resnet backbone output:', k, v)

            resnet_fms=[endpoints['resnet_v2_50/block2'],
                        endpoints['resnet_v2_50/block3'],
                        endpoints['resnet_v2_50/block4']]

            if cfg.MODEL.fpn:
                resnet_fms = create_fpn_net(resnet_fms, dims_list=cfg.MODEL.fpn_dims)


    return resnet_fms

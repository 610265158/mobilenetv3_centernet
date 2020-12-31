import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.net.mobilenetv3 import mobilnet_v3
from lib.core.model.net.mobilenet.mobilenet import training_scope
from lib.core.model.net.mobilenetv3.mobilnet_v3 import hard_swish

def mobilenetv3_large_feature_extractor(image,is_training=True):

    arg_scope = training_scope(weight_decay=cfg.TRAIN.weight_decay_factor, is_training=is_training)

    with tf.contrib.slim.arg_scope(arg_scope):

        _, endpoints = mobilnet_v3.large(image,
                                        depth_multiplier=cfg.MODEL.size,
                                        is_training=is_training,
                                        base_only=True,
                                        finegrain_classification_mode=False)

        for k,v in endpoints.items():
            print('mobile backbone output:',k,v)

        extern_conv = slim.conv2d(_,
                                  480,
                                  [1, 1],
                                  stride=1,
                                  padding='SAME',
                                  activation_fn=hard_swish,
                                  scope='extern1')

        print(extern_conv)
        mobilebet_fms = [endpoints['layer_5/expansion_output'],
                         endpoints['layer_7/expansion_output'],
                         endpoints['layer_13/output'],
                         extern_conv]

    return mobilebet_fms

def mobilenetv3_large_minimalistic_feature_extractor(image,is_training=True):

    arg_scope = training_scope(weight_decay=cfg.TRAIN.weight_decay_factor, is_training=is_training)

    with tf.contrib.slim.arg_scope(arg_scope):

        _, endpoints = mobilnet_v3.large_minimalistic(image,
                                        depth_multiplier=cfg.MODEL.size,
                                        is_training=is_training,
                                        base_only=True,
                                        finegrain_classification_mode=False)

        for k,v in endpoints.items():
            print('mobile backbone output:',k,v)

        extern_conv = slim.conv2d(_,
                                  480,
                                  [1, 1],
                                  stride=1,
                                  padding='SAME',
                                  scope='extern1')


        mobilebet_fms = [endpoints['layer_5/expansion_output'],
                         endpoints['layer_8/expansion_output'],
                         endpoints['layer_13/output'],
                         extern_conv]

    return mobilebet_fms


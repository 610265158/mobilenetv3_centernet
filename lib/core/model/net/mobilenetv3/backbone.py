import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.net.mobilenetv3 import mobilnet_v3
from lib.core.model.net.mobilenet.mobilenet import training_scope
from lib.core.model.net.mobilenetv3.mobilnet_v3 import hard_swish

def mobilenetv3_large_detection(image,is_training=True):

    arg_scope = training_scope(weight_decay=cfg.TRAIN.weight_decay_factor, is_training=is_training)

    with tf.contrib.slim.arg_scope(arg_scope):

        _, endpoints = mobilnet_v3.large(image,
                                        depth_multiplier=0.75,
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
                         endpoints['layer_8/expansion_output'],
                         endpoints['layer_14/expansion_output'],
                         extern_conv]

    return mobilebet_fms


def mobilenetv3_small_minimalistic(image,is_training=True):

    arg_scope = training_scope(weight_decay=cfg.TRAIN.weight_decay_factor, is_training=is_training)

    with tf.contrib.slim.arg_scope(arg_scope):
        if cfg.DATA.channel==1:
            if cfg.MODEL.global_stride==8:
                stride=2
            else:
                stride=1
            image = slim.separable_conv2d(image,
                                          3,
                                          [3, 3],
                                          stride=stride,
                                          padding='SAME',
                                          scope='preconv')
            
        final_feature, endpoints = mobilnet_v3.small_minimalistic(image,
                                        depth_multiplier=1.0,
                                        is_training=is_training,
                                        base_only=True,
                                        finegrain_classification_mode=False)

        extern_conv=slim.separable_conv2d(final_feature, 128,
                                          [3, 3],
                                          stride=2,
                                          padding='SAME',
                                          scope='extern1')
        extern_conv = slim.separable_conv2d(extern_conv, 96,
                                            [3, 3],
                                            padding='SAME',
                                            scope='extern2')
        extern_conv = slim.separable_conv2d(extern_conv, 128,
                                            [3, 3],
                                            padding='SAME',
                                            scope='extern3')


        for k,v in endpoints.items():
            print('mobile backbone output:',k,v)

        mobilebet_fms=[endpoints['layer_3/expansion_output'],
                       endpoints['layer_5/expansion_output'],
                       endpoints['layer_9/expansion_output'],
                       #final_feature,
                       extern_conv]


    return mobilebet_fms

#-*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from train_config import config as cfg

from tensorflow.python.ops.init_ops import Initializer


import numpy as np
import math



class CenternetHead():

    def __call__(self,fms,L2_reg,training=True):

        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):
                c3, c4, c5 = fms


                p5 = slim.conv2d(c5, 256, [1, 1], padding='SAME', scope='C5_reduced')
                p5_upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last')(p5)
                p5_upsampled = slim.conv2d(p5_upsampled, 256, [3, 3], padding='SAME', scope='P5_after')

                p4 = slim.conv2d(c4, 256, [1, 1], padding='SAME', scope='C4_reduced')
                p4 = p4 + p5_upsampled
                p4_upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last')(p4)
                p4_upsampled = slim.conv2d(p4_upsampled, 256, [3, 3], padding='SAME', scope='P4_after')

                p3 = slim.conv2d(c3, 256, [1, 1], padding='SAME', scope='C3_reduced')
                p3 = p3 + p4_upsampled
                p3_upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last',interpolation='bilinear')(p3)


                p2_feature = slim.conv2d(p3_upsampled, 256, [3, 3], padding='SAME', scope='p2_feature')


            size = slim.conv2d(p2_feature,
                                      2,
                                      [3, 3],
                                      stride=1,
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='centernet_reg_output')

            kps = slim.conv2d(p2_feature,
                                cfg.DATA.num_class,
                                [3, 3],
                                stride=1,
                                activation_fn=None,
                                normalizer_fn=None,
                                scope='centernet_cls_output')

        return size,kps






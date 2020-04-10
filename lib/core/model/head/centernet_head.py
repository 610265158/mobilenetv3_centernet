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


                p5_upsampled = self._upsample(c5,scope='upsample1')
                # p4_upsampled = self._upsample(p5_upsampled, scope='upsample2')
                # p2_feature = self._upsample(p4_upsampled, scope='upsample3')

                p4 = tf.concat([c4,p5_upsampled],axis=3)
                p4 = slim.conv2d(p4, 256, [3, 3], padding='SAME', scope='P4_after')

                p3_upsampled = self._upsample(p4,scope='upsample_p4')

                p3 = tf.concat([c3,p3_upsampled],axis=3)
                p3 = slim.conv2d(p3, 128, [3, 3], padding='SAME', scope='P3_after')

                p2_upsampled = self._upsample(p3,scope='upsample_p3')

                p2_feature = slim.conv2d(p2_upsampled, 128, [3, 3], padding='SAME', scope='p2_feature')


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


    def _upsample(self,fm,scope='upsample'):
        upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last')(fm)
        upsampled_conv = slim.conv2d(upsampled, 256, [1, 1], padding='SAME', scope=scope)
        return upsampled_conv


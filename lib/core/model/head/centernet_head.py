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
                deconv_feature=c5
                for i in range(3):
                    deconv_feature=self._upsample(deconv_feature,scope='upsample_%d'%i)
                kps = slim.conv2d(deconv_feature,
                                  64,
                                  [3, 3],
                                  stride=1,
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=None,
                                  biases_initializer=tf.initializers.constant(0),
                                  scope='centernet_pre_cls')
                kps = slim.conv2d(kps,
                                  cfg.DATA.num_class,
                                  [1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant( -2.19),
                                  scope='centernet_cls_output')

                wh = slim.conv2d(deconv_feature,
                                   64,
                                   [3, 3],
                                   stride=1,
                                   activation_fn=tf.nn.relu,
                                   normalizer_fn=None,
                                   biases_initializer=tf.initializers.constant(0),
                                   scope='centernet_pre_wh')
                wh = slim.conv2d(wh,
                                  2,
                                  [1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant(0.),
                                  scope='centernet_wh_output')

                reg = slim.conv2d(deconv_feature,
                                  64,
                                  [3, 3],
                                  stride=1,
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=None,
                                  biases_initializer=tf.initializers.constant(0),
                                  scope='centernet_pre_reg')
                reg = slim.conv2d(reg,
                                 2,
                                 [1, 1],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                 biases_initializer=tf.initializers.constant(0.),
                                 scope='centernet_reg_output')
        return kps,wh,reg


    def _upsample(self,fm,scope='upsample'):
        upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last',interpolation='bilinear')(fm)
        upsampled_conv = slim.conv2d(upsampled, 256, [3, 3], padding='SAME', scope=scope)
        return upsampled_conv
    def _upsample_deconv(self,fm,scope='upsample'):

        upsampled_conv = slim.conv2d_transpose(fm, 256, [4, 4],stride=2, padding='SAME', scope=scope)
        return upsampled_conv
class CenternetHeadLight():

    def __call__(self,fms,L2_reg,training=True):

        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):
                c3, c4, c5 = fms
                deconv_feature=c5
                for i in range(3):
                    deconv_feature=self._upsample(deconv_feature,scope='upsample_%d'%i)


                kps = slim.conv2d(deconv_feature,
                                  cfg.DATA.num_class,
                                  [1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant( -2.19),
                                  scope='centernet_cls_output')


                wh = slim.conv2d(deconv_feature,
                                  2,
                                  [1, 1],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant(0.),
                                  scope='centernet_wh_output')


                reg = slim.conv2d(deconv_feature,
                                 2,
                                 [1, 1],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                 biases_initializer=tf.initializers.constant(0.),
                                 scope='centernet_reg_output')
        return kps,wh,reg


    def _upsample(self,fm,scope='upsample'):
        upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last',interpolation='bilinear')(fm)
        upsampled_conv = slim.separable_conv2d(upsampled, 256, [1, 1], padding='SAME', scope=scope)
        return upsampled_conv
    def _upsample_deconv(self,fm,scope='upsample'):

        upsampled_conv = slim.conv2d_transpose(fm, 256, [4, 4],stride=2, padding='SAME', scope=scope)
        return upsampled_conv
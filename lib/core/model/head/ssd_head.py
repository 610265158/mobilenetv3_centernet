#-*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from train_config import config as cfg

from tensorflow.python.ops.init_ops import Initializer


import numpy as np
import math

class PriorProbability(Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None,partition_info=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=np.float32) *-math.log((1 - self.probability) / self.probability)

        return result


class SSDHead():

    def __call__(self,fms,L2_reg,training=True,ratios_per_pixel=3):

        cla_set=[]
        reg_set=[]
        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('ssdout'):

                for i in range(len(fms)):
                    current_feature = fms[i]

                    dim_h=tf.shape(current_feature)[1]
                    dim_w = tf.shape(current_feature)[2]

                    feature_reg=current_feature

                    for j in range(4):
                        feature_reg = slim.separable_conv2d(feature_reg, 256, [3, 3], stride=1, scope='fpn%d_reg_brach_%d'%(i,j))

                    reg_out = slim.separable_conv2d(feature_reg,
                                          ratios_per_pixel * 4,
                                          [3, 3],
                                          stride=1,
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          scope='fpn%d_reg_output'%i)

                    feature_cls = current_feature
                    for j in range(4):
                        feature_cls = slim.separable_conv2d(feature_cls, 256, [3, 3], stride=1,scope='fpn%d_cls_brach_%d' % (i,j))

                    cls_out = slim.separable_conv2d(feature_cls,
                                          ratios_per_pixel*cfg.DATA.num_class,
                                          [3, 3],
                                          stride=1,
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          weights_initializer=tf.initializers.random_normal(mean=0.0, stddev=0.01,seed=None),
                                          biases_initializer=PriorProbability(),
                                          scope='fpn%dcls_output'%i)


                    reg_out = tf.reshape(reg_out, ([-1, dim_h * dim_w, ratios_per_pixel, 4]))
                    reg_out = tf.reshape(reg_out, ([-1, dim_h * dim_w* ratios_per_pixel, 4]))

                    cls_out = tf.reshape(cls_out, ([-1, dim_h * dim_w, ratios_per_pixel, cfg.DATA.num_class]))
                    cls_out = tf.reshape(cls_out, ([-1, dim_h * dim_w* ratios_per_pixel,cfg.DATA.num_class]))


                    cla_set.append(cls_out)
                    reg_set.append(reg_out)

                reg = tf.concat(reg_set, axis=1)
                cla = tf.concat(cla_set, axis=1)

        return reg,cla






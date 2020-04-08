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

        cla_set=[]
        reg_set=[]
        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):


                current_feature = fms[-1]

                feature_reg=current_feature

                for j in range(3):
                    feature_reg = slim.conv2d_transpose(feature_reg, 256, [3, 3], stride=2, scope='upsample_brach_%d'%(j))


                print(feature_reg)
                reg = slim.separable_conv2d(feature_reg,
                                      2,
                                      [3, 3],
                                      stride=1,
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='centernet_reg_output')

                cls = slim.separable_conv2d(feature_reg,
                                                81,
                                                [3, 3],
                                                stride=1,
                                                activation_fn=None,
                                                normalizer_fn=None,
                                                scope='centernet_cls_output')

        return reg,cls






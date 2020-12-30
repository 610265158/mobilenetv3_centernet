# -*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from train_config import config as cfg

from lib.core.model.sqeeze_excitation.se import se

class CenternetHead():

    def __call__(self, fms, training=True):
        arg_scope = resnet_arg_scope( bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):
                # c2, c3, c4, c5 = fms
                # deconv_feature=c5


                result=[]
                #####
                for i,fm in enumerate(fms):
                    kps = slim.separable_conv2d(fm,
                                      cfg.DATA.num_class,
                                      [3, 3],
                                      stride=1,
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                      biases_initializer=tf.initializers.constant(-2.19),
                                      scope='centernet_cls_output_%d'%i)


                    wh = slim.separable_conv2d(fm,
                                     4,
                                     [3, 3],
                                     stride=1,
                                     activation_fn=None,
                                     normalizer_fn=None,
                                     weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                     biases_initializer=tf.initializers.constant(0),
                                     scope='centernet_wh_output_%d'%i)
                    result.append([kps,wh*16])
        return result


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

                deconv_feature = self._unet_magic(fms)

                #####

                kps = slim.separable_conv2d(deconv_feature,
                                  cfg.DATA.num_class,
                                  [3, 3],
                                  stride=1,
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                  biases_initializer=tf.initializers.constant(-2.19),
                                  scope='centernet_cls_output')


                wh = slim.separable_conv2d(deconv_feature,
                                 4,
                                 [3, 3],
                                 stride=1,
                                 activation_fn=None,
                                 normalizer_fn=None,
                                 weights_initializer=tf.initializers.random_normal(stddev=0.001),
                                 biases_initializer=tf.initializers.constant(0),
                                 scope='centernet_wh_output')

        return kps, wh*16

    def _complex_upsample(self,fm,output_dim, factor=2,scope='upsample'):
        with tf.variable_scope(scope):


            x = slim.separable_conv2d(fm,
                                       output_dim,
                                       [3, 3],
                                       activation_fn=None,
                                       padding='SAME',
                                       scope='branch_x_upsample_resize')
            y = slim.separable_conv2d(fm,
                                       output_dim,
                                       [5, 5],
                                       activation_fn=None,
                                       padding='SAME',
                                       scope='branch_y_upsample_resize')
            final = x+y
            final = tf.keras.layers.UpSampling2D(data_format='channels_last', interpolation='bilinear',
                                                          size=(factor, factor))(final)

            return final

    def revers_conv(self,fm,output_dim,k_size,refraction=4,scope='boring'):

        input_channel = fm.shape[3].value

        mid_channels=input_channel//refraction
        with tf.variable_scope(scope):
            fm_bypass = slim.conv2d(fm,
                             mid_channels,
                             [1, 1],
                             padding='SAME',
                             scope='1x1')

            fm_bypass = slim.separable_conv2d(fm_bypass,
                                              output_dim,
                                              [k_size, k_size],
                                              activation_fn=None,
                                              padding='SAME',
                                              scope='3x3')


            return fm_bypass

    def _unet_magic(self, fms, dims=cfg.MODEL.head_dims):

        c2, c3, c4, c5 = fms

        ####24, 116, 232, 464,

        c5_upsample = self._complex_upsample(c5, output_dim= dims[0]//2,factor=2, scope='c5_upsample')
        c4 = self.revers_conv(c4,  dims[0]//2, k_size=5, scope='c4_reverse')
        p4=tf.nn.relu(tf.concat([c4,c5_upsample],axis=3))

        c4_upsample = self._complex_upsample(p4, output_dim= dims[1]//2, factor=2,scope='c4_upsample')
        c3 = self.revers_conv(c3,  dims[1]//2, k_size=5, scope='c3_reverse')
        p3=tf.nn.relu(tf.concat([c3,c4_upsample],axis=3))

        c3_upsample = self._complex_upsample(p3, output_dim= dims[2]//2,factor=2, scope='c3_upsample')
        c2 = self.revers_conv(c2, dims[2]//2,k_size=5,scope='c2_reverse')
        p2=tf.nn.relu(tf.concat([c2,c3_upsample],axis=3))

        final = se(p2, dims[2])

        return final


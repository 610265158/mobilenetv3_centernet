# -*-coding:utf-8-*-


import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope
from train_config import config as cfg

from lib.core.model.sqeeze_excitation.se import se

class CenternetHead():

    def __call__(self, fms, L2_reg, training=True):
        arg_scope = resnet_arg_scope(weight_decay=L2_reg, bn_is_training=training, )
        with slim.arg_scope(arg_scope):
            with tf.variable_scope('CenternetHead'):
                # c2, c3, c4, c5 = fms
                # deconv_feature=c5

                deconv_feature = self._unet_magic(fms)

                #####

                #kps,wh = self._pre_head(deconv_feature, 'centernet_pre_feature')

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


    def _pre_head(self, fm, scope):

        def _head_conv(fms,dim,child_scope):
            with tf.variable_scope(scope + child_scope):
                x, y= fms

                x = slim.separable_conv2d(x, None, kernel_size=[3, 3], stride=1, activation_fn=None,scope='branchx_3x3_pre')

                y = slim.separable_conv2d(y, None, kernel_size=[5, 5], stride=1, activation_fn=None,scope='branchy_5x5_pre')

            fm = tf.concat([x,y], axis=3)

            return fm

        split_fm = tf.split(fm, num_or_size_splits=2, axis=3)

        kps = _head_conv(split_fm,dim=96,child_scope='kps')

        wh  = _head_conv(split_fm, dim=48, child_scope='wh')

        return kps,wh

    def _complex_upsample(self,fm,input_dim,output_dim,use_se=False, factor=2,scope='upsample'):
        with tf.variable_scope(scope):
            x = fm[:, :, :, :input_dim // 2]
            y = fm[:, :, :, input_dim // 2:]

            x = self._upsample_resize(x, dim=output_dim // 2, k_size=3,factor=factor, scope='branch_x_upsample_resize')

            y = self._upsample_group_deconv(y, dim=output_dim // 2, group=2,factor=factor, scope='branch_y_upsample_deconv')

            final = tf.concat([x, y],axis=3)

            if use_se:

                final=se(final,output_dim)


            return final

    def _upsample_resize(self, fm, k_size=5, dim=256, factor=2,scope='upsample'):

        upsampled_conv = slim.separable_conv2d(fm,
                                               dim,
                                               [k_size, k_size],
                                               padding='SAME',
                                               scope=scope)

        upsampled_conv = tf.keras.layers.UpSampling2D(data_format='channels_last',size=(factor,factor))(upsampled_conv)

        return upsampled_conv

    def _upsample_group_deconv(self, fm,dim,group=4,factor=2, scope='upsample'):
        '''
        group deconv

        :param fm: input feature
        :param dim: input dim , should be n*group
        :param group:
        :param scope:
        :return:
        '''
        sliced_fms=tf.split(fm, num_or_size_splits=group, axis=3)

        deconv_fms=[]
        for i in range(group):
            cur_upsampled_conv = slim.conv2d_transpose(sliced_fms[i],
                                                       dim//group,
                                                       [4, 4],
                                                       stride=2,
                                                       padding='SAME',
                                                       scope=scope+'group_%d'%i)
            deconv_fms.append(cur_upsampled_conv)

        deconv_fm= tf.concat(deconv_fms, axis=3)


        if factor//2!=1:
            deconv_fm = tf.keras.layers.UpSampling2D(data_format='channels_last', size=(factor//2, factor//2))(
                deconv_fm)

        return deconv_fm

    def _unet_magic(self, fms, dims=cfg.MODEL.head_dims):

        c2, c3, c4, c5 = fms

        ####24, 116, 232, 464,

        input_channel=c5.shape[3].value
        c5_upsample = self._complex_upsample(c5, input_dim=input_channel, output_dim=dims[0]//3*2,factor=2, scope='c5_upsample')
        c4 = slim.separable_conv2d(c4,
                         dims[0]//3,
                         [3, 3],
                         padding='SAME',
                         scope='c4_1x1')

        p4=tf.concat([c4,c5_upsample],axis=3)
        p4=self._shuffle(p4,3)




        print('xxx',p4)
        c4_upsample = self._complex_upsample(p4, input_dim=dims[0], output_dim=dims[1]//3*2, factor=2,scope='c4_upsample')
        c3 = slim.separable_conv2d(c3,
                         dims[1]//3,
                         [3, 3],
                         padding='SAME',
                         scope='c3_1x1')

        p3 = tf.concat([c3, c4_upsample], axis=3)

        print('yyy', p3)
        p3 = self._shuffle(p3, 3)



        print('yyy',p3)

        c3_upsample = self._complex_upsample(p3, input_dim=dims[1], output_dim=dims[2]//3*2,factor=2, scope='c3_upsample')
        c2 = slim.separable_conv2d(c2,
                         dims[2]//3,
                         [3, 3],
                         padding='SAME',
                         scope='c2_1x1')

        p2 = tf.concat([c2, c3_upsample], axis=3)

        return p2

    def _shuffle(self,z,group=2):

        with tf.name_scope('shuffle'):
            shape = tf.shape(z)
            batch_size = shape[0]
            height, width = shape[1], shape[2]

            depth = z.shape[3].value//group

            if cfg.MODEL.deployee:
                z = tf.reshape(z, [height, width, group, depth])  # shape [batch_size, height, width, 2, depth]

                z = tf.transpose(z, [0, 1, 3, 2])

            else:
                z = tf.reshape(z,
                               [batch_size, height, width, group, depth])  # shape [batch_size, height, width, 2, depth]

                z = tf.transpose(z, [0, 1, 2, 4, 3])

            z = tf.reshape(z, [batch_size, height, width, group * depth])

            return z

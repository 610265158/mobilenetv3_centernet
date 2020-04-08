import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.net.mobilenet.mobilenet_v2 import mobilenet_v2_050,mobilenet_v2_035,mobilenet_v2_025
from lib.core.model.net.mobilenet.mobilenet import training_scope


from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope




def create_fpn_net(blocks,dims_list):

    of1, of2, of3= blocks

    # lateral2 = slim.conv2d(of2, dims_list[1], [1, 1],
    #                       padding='SAME',
    #                       scope='lateral/res{}'.format(2))
    #
    # upsample2_of3 = slim.conv2d(of3, dims_list[1], [1, 1],
    #                        padding='SAME',
    #                        scope='merge/res{}'.format(2))
    # upsample2 = tf.keras.layers.UpSampling2D(data_format='channels_last' )(upsample2_of3)

    # fem_2 = lateral2 + upsample2

    lateral1 = slim.conv2d(of1, dims_list[0], [1, 1],
                           padding='SAME',
                           scope='lateral/res{}'.format(1))

    upsample1_of2 = slim.conv2d(of2, dims_list[0], [1, 1],
                            padding='SAME',
                            scope='merge/res{}'.format(1))
    upsample1 = tf.keras.layers.UpSampling2D(data_format='channels_last')(upsample1_of2)

    fem_1 = lateral1 + upsample1

    #####enhance model
    fpn_fms = [fem_1, upsample1_of2, of3]

    return fpn_fms

def mobilenet_ssd(image,L2_reg,is_training=True):

    arg_scope = training_scope(weight_decay=L2_reg, is_training=is_training)

    with tf.contrib.slim.arg_scope(arg_scope):
        _,endpoint = mobilenet_v2_035(image,is_training=is_training,base_only=True,finegrain_classification_mode=False)

        for k,v in endpoint.items():
            print('mobile backbone output:',k,v)

        mobilebet_fms=[
                       endpoint['layer_8/expansion_output'],
                       endpoint['layer_15/expansion_output'],
                       endpoint['layer_18/output']]

        if cfg.MODEL.fpn:
            mobilebet_fms=create_fpn_net(mobilebet_fms,dims_list=cfg.MODEL.fpn_dims)

    return mobilebet_fms

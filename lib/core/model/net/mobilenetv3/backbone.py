import tensorflow as tf
import tensorflow.contrib.slim as slim

from train_config import config as cfg

from lib.core.model.net.mobilenetv3 import mobilnet_v3
from lib.core.model.net.mobilenet.mobilenet import training_scope

from lib.core.model.fpn.seperateconv_fpn import create_fpn_net

def mobilenetv3_ssd(image,is_training=True):

    arg_scope = training_scope(weight_decay=cfg.TRAIN.weight_decay_factor, is_training=is_training)

    with tf.contrib.slim.arg_scope(arg_scope):

        _, endpoints = mobilnet_v3.large(image,
                                        depth_multiplier=1.,
                                        is_training=is_training,
                                        base_only=True,
                                        finegrain_classification_mode=False)


        end_feature=slim.separable_conv2d(endpoints['layer_16/output'],
                                          512,
                                          [3, 3],
                                          stride=1,
                                          scope='mbntev3_extra_conv')

        # for k,v in endpoints.items():
        #     print('mobile backbone output:',k,v)
        #
        # mobilebet_fms=[endpoints['layer_8/expansion_output'],
        #                endpoints['layer_14/expansion_output'],
        #                end_feature]
        #
        # if cfg.MODEL.fpn:
        #     mobilebet_fms=create_fpn_net(mobilebet_fms,dims_list=cfg.MODEL.fpn_dims)

    return [end_feature]

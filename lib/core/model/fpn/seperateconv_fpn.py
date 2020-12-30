#-*-coding:utf-8-*-

import tensorflow as tf
import tensorflow.contrib.slim as slim


def create_fpn_net(blocks,dims_list=[96,96,96]):

    c3, c4, c5= blocks

    ## up-down
    c3_p=slim.conv2d(c3, dims_list[0], [1, 1],padding='SAME',scope='C3p_reduced')

    c3_p_downsample = slim.separable_conv2d(c3_p, dims_list[0], [3, 3],stride=2, padding='SAME',scope='C3p_downsample')
    c4_p = slim.conv2d(c4, dims_list[0], [1, 1], padding='SAME', scope='C4p_reduced')
    c4_p = c4_p+c3_p_downsample

    c4_p_downsample = slim.separable_conv2d(c4_p, dims_list[0], [3, 3], stride=2,padding='SAME', scope='C4p_downsample')
    c5_p = slim.conv2d(c5, dims_list[0], [1, 1], padding='SAME', scope='C5p_reduced')
    c5_p = c5_p+c4_p_downsample


    ##bottom-up
    c3, c4, c5= c3_p,c4_p,c5_p
    p5 = slim.separable_conv2d(c5, dims_list[2], [3, 3],padding='SAME',scope='p5_feature')

    p5_upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last')(p5)
    p4 = slim.separable_conv2d(c4, dims_list[1], [3, 3],padding='SAME',scope='C4_reduced')
    p4 = p4 + p5_upsampled
    p4 = slim.separable_conv2d(p4, dims_list[2], [3, 3], padding='SAME', scope='p4_feature')

    p4_upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last')(p4)
    p3 = c3
    p3 = p3 + p4_upsampled
    p3 = slim.separable_conv2d(p3, dims_list[2], [3, 3], padding='SAME', scope='p3_feature')

    # p6 = slim.separable_conv2d(c5,  dims_list[3], [3, 3], stride=2, scope='p6')
    # p7 = slim.separable_conv2d(p6,  dims_list[4], [3, 3], stride=2, scope='p7')



    fpn_fms = [p3,p4,p5]
    for fm in fpn_fms:
        print(fm)
    return fpn_fms
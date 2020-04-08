#-*-coding:utf-8-*-

import tensorflow as tf
import tensorflow.contrib.slim as slim


def create_fpn_net(blocks,dims_list):

    c3, c4, c5= blocks

    p5 = slim.conv2d(c5, dims_list[2], [1, 1],padding='SAME',scope='C5_reduced')
    p5_upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last')(p5)
    p5 = slim.conv2d(p5, dims_list[1], [3, 3],padding='SAME',scope='P5')

    p4 = slim.conv2d(c4, dims_list[1], [1, 1],padding='SAME',scope='C4_reduced')
    p4 = p4 + p5_upsampled
    p4_upsampled = tf.keras.layers.UpSampling2D(data_format='channels_last')(p4)
    p4 = slim.conv2d(p4, dims_list[1], [3, 3],padding='SAME',scope='P4')

    p3 = slim.conv2d(c3, dims_list[0], [1, 1], padding='SAME', scope='C3_reduced')
    p3 = p3 + p4_upsampled
    p3 = slim.conv2d(p3, dims_list[1], [3, 3], padding='SAME', scope='P3')

    p6 = slim.conv2d(c5,  dims_list[3], [3, 3], stride=2, scope='p6')
    p7 = slim.conv2d(p6,  dims_list[4], [3, 3], stride=2, scope='p7')

    fpn_fms = [p3,p4,p5,p6,p7]
    for fm in fpn_fms:
        print(fm)
    return fpn_fms
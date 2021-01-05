import tensorflow as tf
import tensorflow.contrib.slim as slim

def se(fm,input_dim,refraction=4):

    se = slim.separable_conv2d(fm,
                     input_dim//refraction,
                     [1, 1],
                     stride=1,
                     scope='conv1x1_se_a')
    se = slim.conv2d(se,
                     input_dim,
                     [1, 1],
                     stride=1,
                     activation_fn=None,
                     scope='conv1x1_se_b')



    return fm*se
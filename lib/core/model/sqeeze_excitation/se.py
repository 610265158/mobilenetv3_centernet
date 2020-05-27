import tensorflow as tf
import tensorflow.contrib.slim as slim

def se(fm,input_dim,refraction=4):
    se=tf.reduce_mean(fm,axis=[1,2],keep_dims=True)
    se = slim.conv2d(se,
                     input_dim//refraction,
                     [1, 1],
                     stride=1,
                     activation_fn=tf.nn.relu,
                     biases_initializer=None,
                     normalizer_fn=slim.batch_norm,
                     scope='conv1x1_se_a')
    se = slim.conv2d(se,
                     input_dim,
                     [1, 1],
                     stride=1,
                     activation_fn=None,
                     normalizer_fn=None,
                     biases_initializer=None,
                     scope='conv1x1_se_b')

    se=tf.nn.sigmoid(se)

    return fm*se
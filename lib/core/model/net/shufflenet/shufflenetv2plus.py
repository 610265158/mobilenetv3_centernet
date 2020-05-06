#-*-coding:utf-8-*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from train_config import config as cfg
from lib.core.model.net.mobilenetv3.mobilnet_v3 import hard_swish




def se(fm,input_dim):
    se=tf.reduce_mean(fm,axis=[1,2],keep_dims=True)
    se = slim.conv2d(se,
                    input_dim//4,
                    [1, 1],
                    stride=1,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm, scope='conv1x1_se')
    se = slim.conv2d(se,
                    input_dim,
                    [1, 1],
                    stride=1,
                    activation_fn=None,
                    normalizer_fn=None, scope='conv1x1_se_a')

    se=tf.nn.relu6(se+3)/6

    return fm*se


def shuffle(z):
    with tf.name_scope('shuffle_split'):
        shape = tf.shape(z)
        batch_size = shape[0]
        height, width = shape[1], shape[2]

        depth = z.shape[3].value

        if cfg.MODEL.deployee:
            z = tf.reshape(z, [ height, width, depth//2, depth])  # shape [batch_size, height, width, 2, depth]

            z = tf.transpose(z, [0, 1, 3, 2])

        else:
            z = tf.reshape(z, [batch_size, height, width, 2, depth//2])# shape [batch_size, height, width, 2, depth]

            z = tf.transpose(z, [0, 1, 2, 4, 3])

        z = tf.reshape(z, [batch_size, height, width, depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y

def shufflenet(old_x,inp, oup, base_mid_channels, *, ksize, stride, activation, useSE):
    x_proj, x = shuffle(old_x)
    base_mid_channel = base_mid_channels
    ksize = ksize
    ksize // 2

    if activation == 'ReLU':
        act_func=tf.nn.relu
    else:
        act_func = hard_swish
    ##branch main
    x = slim.conv2d(x,
                    base_mid_channel,
                    [1, 1],
                    stride=1,
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm, scope='conv1x1_before')

    x = slim.separable_conv2d(x,
                              num_outputs=None,
                              kernel_size=[ksize, ksize],
                              stride=stride,
                              activation_fn=act_func,
                              normalizer_fn=None,
                              scope='conv_dp_1')

    x = slim.conv2d(x,
                    num_outputs=oup,
                    kernel_size=[1, 1],
                    stride=1,
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm,
                    scope='conv1x1_pw')
    if useSE:
        x=se(x,oup)

    if stride == 2:
        x_proj = slim.separable_conv2d(x_proj,
                                  num_outputs=None,
                                  kernel_size=[ksize, ksize],
                                  stride=stride,
                                  activation_fn=None,
                                  normalizer_fn=slim.batch_norm,
                                  scope='conv_dp_proj')

        x_proj = slim.conv2d(x_proj,
                  num_outputs=oup,
                  kernel_size=[1, 1],
                  stride=1,
                  activation_fn=act_func,
                  normalizer_fn=slim.batch_norm,
                  scope='conv1x1_pw_proj')



    res=tf.concat([x,x_proj],axis=3)

    return res

def shufflenet_xception(old_x,inp, oup, base_mid_channels, *, ksize, stride, activation, useSE):
    x_proj, x = shuffle(old_x)
    base_mid_channel = base_mid_channels
    ksize = ksize
    ksize // 2

    if activation == 'ReLU':
        act_func=tf.nn.relu
    else:
        act_func = hard_swish
    ##branch main
    x = slim.conv2d(x,
                    inp,
                    [3, 3],
                    stride=1,
                    activation_fn=None,
                    normalizer_fn=slim.batch_norm, scope='conv1x1_first')

    x = slim.conv2d(x,
                    base_mid_channel,
                    [1, 1],
                    stride=1,
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm, scope='conv1x1_before')
    x = slim.separable_conv2d(x,
                              num_outputs=None,
                              kernel_size=[3, 3],
                              stride=stride,
                              activation_fn=act_func,
                              normalizer_fn=None,
                              scope='conv_dp_1')

    x = slim.conv2d(x,
                    num_outputs=base_mid_channel,
                    kernel_size=[1, 1],
                    stride=1,
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm,
                    scope='conv1x1_pw')

    x = slim.separable_conv2d(x,
                              num_outputs=None,
                              kernel_size=[3, 3],
                              stride=stride,
                              activation_fn=act_func,
                              normalizer_fn=None,
                              scope='conv_dp_1')
    x = slim.conv2d(x,
                    num_outputs=oup,
                    kernel_size=[1, 1],
                    stride=1,
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm,
                    scope='conv1x1_pw')
    if useSE:
        x = se(x, oup)


    if stride == 2:
        x_proj = slim.separable_conv2d(x_proj,
                                  num_outputs=None,
                                  kernel_size=[ksize, ksize],
                                  stride=stride,
                                  activation_fn=None,
                                  normalizer_fn=slim.batch_norm,
                                  scope='conv_dp_proj')

        x_proj = slim.conv2d(x_proj,
                  num_outputs=oup,
                  kernel_size=[1, 1],
                  stride=1,
                  activation_fn=act_func,
                  normalizer_fn=slim.batch_norm,
                  scope='conv1x1_pw_proj')



    res=tf.concat([x,x_proj],axis=3)

    return res




def shufflenet_arg_scope(weight_decay=cfg.TRAIN.weight_decay_factor,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     use_batch_norm=True,
                     batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
  """Defines the default ResNet arg scope.
  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': batch_norm_updates_collections,
      'fused': True,  # Use fused batch norm if possible.
  }

  with slim.arg_scope(
      [slim.conv2d,slim.separable_conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      biases_initializer=None,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc








def ShufflenetV2Plus(inputs,is_training=True,model_size='Small'):

    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]

    stage_repeats = [4, 4, 8, 4]
    if model_size == 'Large':
        stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
    elif model_size == 'Medium':
        stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
    elif model_size == 'Small':
        stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
    else:
        raise NotImplementedError

    arg_scope = shufflenet_arg_scope(weight_decay=cfg.TRAIN.weight_decay_factor)
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with tf.variable_scope('ShuffleNetV2Plus'):

                net = slim.conv2d(inputs, 16, [3, 3],stride=2, activation_fn=hard_swish,
                                  normalizer_fn=slim.batch_norm, scope='init_conv')




    return




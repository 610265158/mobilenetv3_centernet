#-*-coding:utf-8-*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from train_config import config as cfg


def torch_style_padding(inputs,kernel_size,rate=1):

    '''
    by default tensorflow use different padding method with pytorch,
    so we need do explicit padding before we do conv or pool
    :param inputs:
    :param kernel_size:
    :param rate:
    :return:
    '''
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

    return inputs

def shuffle(z, torch_style_shuffle=False):

    if not torch_style_shuffle:
        with tf.name_scope('shuffle_split'):
            shape = tf.shape(z)
            batch_size = shape[0]
            height, width = z.shape[1].value, z.shape[2].value

            depth = z.shape[3].value

            if cfg.MODEL.deployee:
                z = tf.reshape(z, [ height, width, 2, depth//2])  # shape [batch_size, height, width, 2, depth]

                z = tf.transpose(z, [0, 1, 3, 2])

            else:
                z = tf.reshape(z, [batch_size, height, width, 2, depth//2])# shape [batch_size, height, width, 2, depth]

                z = tf.transpose(z, [0, 1, 2, 4, 3])

            z = tf.reshape(z, [batch_size, height, width, depth])
            x, y = tf.split(z, num_or_size_splits=2, axis=3)
            return x, y
    else:
        with tf.name_scope('shuffle_split'):

            z=tf.transpose(z,perm=[0,3,1,2])

            shape = tf.shape(z)
            batch_size = shape[0]
            height, width = z.shape[2].value, z.shape[3].value

            depth = z.shape[1].value

            if cfg.MODEL.deployee:
                z = tf.reshape(z,[batch_size * depth // 2, 2, height * width])  # shape [batch_size, height, width, 2, depth]

                z = tf.transpose(z, [1, 0, 2])
                z = tf.reshape(z, [batch_size*2,  depth // 2, height, width])

                z = tf.transpose(z, perm=[0, 2, 3, 1])

                x, y = tf.split(z, num_or_size_splits=2, axis=0)


            else:
                z = tf.reshape(z, [batch_size*depth//2,2, height* width])# shape [batch_size, height, width, 2, depth]

                z = tf.transpose(z, [1,0,2])
                z = tf.reshape(z, [batch_size*2, depth // 2, height , width])
                z = tf.transpose(z, perm=[0, 2, 3, 1])
                x, y = tf.split(z, num_or_size_splits=2, axis=0)




            return x, y

def ShuffleV2Block(old_x,inp, oup, base_mid_channels, ksize, stride,scope_index=0):




    main_scope_list=[['0','3','5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],
                     ['0', '3', '5'],

                     ]



    project_scope_list=[['0','2'],
                        None,
                        None,
                        None,
                        ['0', '2'],
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        ['0', '2'],   #16
                        None,
                        None,
                        None,
                        ]




    main_scope=main_scope_list[scope_index]
    project_scope = project_scope_list[scope_index]


    if stride==1:
        x_proj, x = shuffle(old_x)
    else:
        x_proj = old_x
        x = old_x

    base_mid_channel = base_mid_channels

    outputs = oup - inp



    act_func=tf.nn.relu

    ##branch main

    x = slim.conv2d(x,
                    base_mid_channel,
                    [1, 1],
                    stride=1,
                    padding='VALID',
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm,
                    biases_initializer=None,
                    scope='branch_main/'+main_scope[0])
    x = torch_style_padding(x, ksize)
    x = slim.separable_conv2d(x,
                              num_outputs=None,
                              kernel_size=[ksize, ksize],
                              stride=stride,
                              padding='VALID',
                              activation_fn=None,
                              normalizer_fn=slim.batch_norm,
                              scope='branch_main/'+main_scope[1])

    x = slim.conv2d(x,
                    num_outputs=outputs,
                    kernel_size=[1, 1],
                    stride=1,
                    padding='VALID',
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm,
                    scope='branch_main/'+main_scope[2])


    if stride == 2:
        x_proj = torch_style_padding(x_proj, ksize)
        x_proj = slim.separable_conv2d(x_proj,
                                       num_outputs=None,
                                       kernel_size=[ksize, ksize],
                                       stride=stride,
                                       padding='VALID',
                                       activation_fn=None,
                                       normalizer_fn=slim.batch_norm,
                                       scope='branch_proj/'+project_scope[0])

        x_proj = slim.conv2d(x_proj,
                             num_outputs=inp,
                             kernel_size=[1, 1],
                             stride=1,
                             padding='VALID',
                             activation_fn=act_func,
                             normalizer_fn=slim.batch_norm,
                             scope='branch_proj/'+project_scope[1])


    res=tf.concat([x_proj,x],axis=3)

    return res


def shufflenet_arg_scope(weight_decay=cfg.TRAIN.weight_decay_factor,
                     batch_norm_decay=0.97,
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








def ShuffleNetV2_5x5(inputs,is_training=True,model_size=cfg.MODEL.size,include_head=False):
    stage_repeats = [4, 8, 4]
    model_size = model_size
    if model_size == '0.5x':
        stage_out_channels = [-1, 24, 48, 96, 192, 1024]
    elif model_size == '1.0x':
        stage_out_channels = [-1, 24, 116, 232, 464, 1024]
    elif model_size == '1.5x':
        stage_out_channels = [-1, 24, 176, 352, 704, 1024]
    elif model_size == '2.0x':
        stage_out_channels = [-1, 24, 244, 488, 976, 2048]
    else:
        raise NotImplementedError

    fms = []
    arg_scope = shufflenet_arg_scope(weight_decay=cfg.TRAIN.weight_decay_factor)
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            with tf.variable_scope('ShuffleNetV2_5x5'):
                input_channel = stage_out_channels[1]


                inputs=torch_style_padding(inputs,3)
                net = slim.conv2d(inputs,
                                  24,
                                  [3, 3],
                                  stride=2,
                                  padding='VALID',
                                  activation_fn=tf.nn.relu,
                                  normalizer_fn=slim.batch_norm,
                                  scope='first_conv/0')
                net = torch_style_padding(net, 3)

                net = slim.max_pool2d(net,kernel_size=3,stride=2,padding='VALID')

                fms = [net]

                feature_cnt=0
                for idxstage in range(len(stage_repeats)):
                    numrepeat = stage_repeats[idxstage]
                    output_channel = stage_out_channels[idxstage + 2]

                    for i in range(numrepeat):
                        with tf.variable_scope('features/%d' % (feature_cnt)):
                            if i == 0:
                                net=ShuffleV2Block(net,input_channel, output_channel,
                                                                    base_mid_channels=output_channel // 2, ksize=5, stride=2,scope_index=feature_cnt)
                            else:
                                net=ShuffleV2Block(net,input_channel // 2, output_channel,
                                                                    base_mid_channels=output_channel // 2, ksize=5, stride=1,scope_index=feature_cnt)

                        input_channel = output_channel
                        feature_cnt+=1
                    fms.append(net)

                if not include_head:
                    return fms


                if include_head:
                    x = slim.conv2d(net,
                                    num_outputs=stage_out_channels[-1],
                                    kernel_size=[1, 1],
                                    stride=1,
                                    padding='VALID',
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=slim.batch_norm,
                                    scope='conv_last/0')

                    x=tf.reduce_mean(x,axis=[1,2],keep_dims=True)

                    if model_size=='2.0x':
                        x=slim.dropout(x,0.8)

                    x=slim.conv2d(x,
                                  num_outputs=cfg.MODEL.cls,
                                  kernel_size=[1, 1],
                                  stride=1,
                                  padding='VALID',
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='classifier/0')

        x = tf.squeeze(x, axis=1)
        x = tf.squeeze(x, axis=1)
        x = tf.identity(x,name='cls_output')
    return x



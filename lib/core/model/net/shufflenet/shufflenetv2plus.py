#-*-coding:utf-8-*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
from train_config import config as cfg
from lib.core.model.net.mobilenetv3.mobilnet_v3 import hard_swish




def se(fm,input_dim,scope='really_boring'):
    se=tf.reduce_mean(fm,axis=[1,2],keep_dims=True)

    with tf.variable_scope(scope):
        se = slim.conv2d(se,
                         input_dim//4,
                         [1, 1],
                         stride=1,
                         activation_fn=tf.nn.relu,
                         biases_initializer=None,
                         normalizer_fn=slim.batch_norm,
                         scope='SE_opr/1')
        se = slim.conv2d(se,
                         input_dim,
                         [1, 1],
                         stride=1,
                         activation_fn=None,
                         normalizer_fn=None,
                         biases_initializer=None,
                         scope='SE_opr/4')

    se=tf.nn.relu6(se+3.)/6.

    return fm*se


def shuffle(z):
    # with tf.name_scope('shuffle_split'):
    #     shape = tf.shape(z)
    #     batch_size = shape[0]
    #     height, width = z.shape[1].value, z.shape[2].value
    #
    #     depth = z.shape[3].value
    #
    #     if cfg.MODEL.deployee:
    #         z = tf.reshape(z, [ height, width, 2, depth//2])  # shape [batch_size, height, width, 2, depth]
    #
    #         z = tf.transpose(z, [0, 1, 3, 2])
    #
    #     else:
    #         z = tf.reshape(z, [batch_size, height, width, 2, depth//2])# shape [batch_size, height, width, 2, depth]
    #
    #         z = tf.transpose(z, [0, 1, 2, 4, 3])
    #
    #     z = tf.reshape(z, [batch_size, height, width, depth])
    #     x, y = tf.split(z, num_or_size_splits=2, axis=3)
    #     return x, y
    with tf.name_scope('shuffle_split'):
        z=tf.transpose(z,perm=[0,3,1,2])

        shape = tf.shape(z)
        batch_size = shape[0]
        height, width = z.shape[2].value, z.shape[3].value

        depth = z.shape[1].value

        if cfg.MODEL.deployee:
            z = tf.reshape(z, [ height, width, 2, depth//2])  # shape [batch_size, height, width, 2, depth]

            z = tf.transpose(z, [0, 1, 3, 2])

        else:
            z = tf.reshape(z, [batch_size*depth//2,2, height* width])# shape [batch_size, height, width, 2, depth]

            z = tf.transpose(z, [1,0,2])
            z = tf.reshape(z, [2,-1, depth // 2, height , width])

        x, y = tf.split(z, num_or_size_splits=2, axis=0)


        x=tf.transpose(x[0],perm=[0,2,3,1])
        y = tf.transpose(y[0], perm=[0, 2, 3, 1])

        return x, y
def shufflenet(old_x,inp, oup, base_mid_channels, ksize, stride, activation, useSE,scope_index=0):




    main_scope_list=[['0','3','5'],
                     ['0', '3', '5'],
                     None,
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
                     None,
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
                        ['0', '2'],
                        None,
                        None,         #10
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


    se_scope_list=[None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None,
                   ['8'],
                   ['8'],
                   ['8'],
                   ['8'],
                   ['8'],
                   ['8'],
                   ['8'],
                   ['8'],
                   ['8'],
                   ['8'],
                   None,
                   ['8'],
                   ]
    print('excuted here')
    main_scope=main_scope_list[scope_index]
    project_scope = project_scope_list[scope_index]
    se_scope=se_scope_list[scope_index]


    if stride==1:
        x_proj, x = shuffle(old_x)
    else:
        x_proj = old_x
        x = old_x

    base_mid_channel = base_mid_channels

    outputs = oup - inp


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
                    normalizer_fn=slim.batch_norm,
                    biases_initializer=None,
                    scope='branch_main/'+main_scope[0])

    x = slim.separable_conv2d(x,
                              num_outputs=None,
                              kernel_size=[ksize, ksize],
                              stride=stride,
                              activation_fn=None,
                              normalizer_fn=slim.batch_norm,
                              scope='branch_main/'+main_scope[1])

    x = slim.conv2d(x,
                    num_outputs=outputs,
                    kernel_size=[1, 1],
                    stride=1,
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm,
                    scope='branch_main/'+main_scope[2])

    if useSE and activation != 'ReLU':
        x=se(x,outputs,scope='branch_main/'+se_scope[0])

    if stride == 2:
        x_proj = slim.separable_conv2d(x_proj,
                                  num_outputs=None,
                                  kernel_size=[ksize, ksize],
                                  stride=stride,
                                  activation_fn=None,
                                  normalizer_fn=slim.batch_norm,
                                  scope='branch_proj/'+project_scope[0])

        x_proj = slim.conv2d(x_proj,
                  num_outputs=inp,
                  kernel_size=[1, 1],
                  stride=1,
                  activation_fn=act_func,
                  normalizer_fn=slim.batch_norm,
                  scope='branch_proj/'+project_scope[1])


    res=tf.concat([x_proj,x],axis=3)

    return res

def shufflenet_xception(old_x,inp, oup, base_mid_channels, stride, activation, useSE,scope_index=0):
    main_scope_list = [None,
                       None,
                       ['0', '2', '5','7','10','12'],
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       None,
                       ['0', '2', '5','7','10','12'],
                       ]

    project_scope_list = [None,
                           None,
                          ['0', '2'],
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,
                          None,

                          ]

    se_scope_list = [None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     None,
                     ['15'],

                     ]

    main_scope = main_scope_list[scope_index]
    project_scope = project_scope_list[scope_index]
    se_scope = se_scope_list[scope_index]


    print(se_scope)


    print(scope_index)
    if stride == 1:
        x_proj, x = shuffle(old_x)
    else:
        x_proj = old_x
        x = old_x

    base_mid_channel = base_mid_channels
    outputs = oup - inp
    if activation == 'ReLU':
        act_func=tf.nn.relu
    else:
        act_func = hard_swish
    ##branch main



    x = slim.separable_conv2d(x,
                              num_outputs=None,
                              kernel_size=[3, 3],
                              stride=stride,
                              activation_fn=None,
                              normalizer_fn=slim.batch_norm,
                              scope='branch_main/'+main_scope[0])

    x = slim.conv2d(x,
                    base_mid_channel,
                    [1, 1],
                    stride=1,
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm,
                    scope='branch_main/'+main_scope[1])

    x = slim.separable_conv2d(x,
                              num_outputs=None,
                              kernel_size=[3, 3],
                              stride=stride,
                              activation_fn=None,
                              normalizer_fn=slim.batch_norm,
                              scope='branch_main/'+main_scope[2])

    x = slim.conv2d(x,
                    num_outputs=base_mid_channel,
                    kernel_size=[1, 1],
                    stride=1,
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm,
                    scope='branch_main/'+main_scope[3])

    x = slim.separable_conv2d(x,
                              num_outputs=None,
                              kernel_size=[3, 3],
                              stride=stride,
                              activation_fn=None,
                              normalizer_fn=slim.batch_norm,
                              scope='branch_main/'+main_scope[4])
    x = slim.conv2d(x,
                    num_outputs=outputs,
                    kernel_size=[1, 1],
                    stride=1,
                    activation_fn=act_func,
                    normalizer_fn=slim.batch_norm,
                    scope='branch_main/'+main_scope[5])
    if useSE and activation != 'ReLU':
        x = se(x, outputs,scope='branch_main/'+se_scope[0])


    if stride == 2:
        x_proj = slim.separable_conv2d(x_proj,
                                      num_outputs=None,
                                      kernel_size=[3, 3],
                                      stride=stride,
                                      activation_fn=None,
                                      normalizer_fn=slim.batch_norm,
                                      scope='conv_dp_proj')

        x_proj = slim.conv2d(x_proj,
                  num_outputs=inp,
                  kernel_size=[1, 1],
                  stride=1,
                  activation_fn=act_func,
                  normalizer_fn=slim.batch_norm,
                  scope='conv1x1_pw_proj')

    res=tf.concat([x_proj,x],axis=3)

    return res




def shufflenet_arg_scope(weight_decay=cfg.TRAIN.weight_decay_factor,
                     batch_norm_decay=0.9,
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
      normalizer_params=batch_norm_params,):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc








def ShufflenetV2Plus(inputs,is_training=True,model_size='Small',include_head=False):

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


    fms=[]
    arg_scope = shufflenet_arg_scope(weight_decay=cfg.TRAIN.weight_decay_factor)
    with slim.arg_scope(arg_scope):
        with slim.arg_scope([slim.batch_norm,slim.dropout], is_training=is_training):
            with tf.variable_scope('ShuffleNetV2_Plus'):
                input_channel = stage_out_channels[1]

                net = slim.conv2d(inputs, 16, [3, 3],stride=2, activation_fn=hard_swish,
                                  normalizer_fn=slim.batch_norm, scope='first_conv/0')

                archIndex=0

                feature_cnt=0
                for idxstage in range(len(stage_repeats)):

                    numrepeat = stage_repeats[idxstage]
                    output_channel = stage_out_channels[idxstage + 2]

                    activation = 'HS' if idxstage >= 1 else 'ReLU'
                    useSE = 'True' if idxstage >= 2 else False
                    for i in range(numrepeat):

                        with tf.variable_scope('features/%d'%(feature_cnt)):
                            if i == 0:
                                inp, outp, stride = input_channel, output_channel, 2
                            else:
                                inp, outp, stride = input_channel // 2, output_channel, 1

                            blockIndex = architecture[archIndex]
                            archIndex += 1
                            if blockIndex == 0:
                                print('Shuffle3x3')
                                net=shufflenet(net,inp, outp, base_mid_channels=outp // 2, ksize=3, stride=stride,
                                               activation=activation, useSE=useSE,scope_index=feature_cnt)
                            elif blockIndex == 1:
                                print('Shuffle5x5')
                                net =shufflenet(net,inp, outp, base_mid_channels=outp // 2, ksize=5, stride=stride,
                                               activation=activation, useSE=useSE,scope_index=feature_cnt)
                            elif blockIndex == 2:
                                print('Shuffle7x7')
                                net=shufflenet(net,inp, outp, base_mid_channels=outp // 2, ksize=7, stride=stride,
                                               activation=activation, useSE=useSE,scope_index=feature_cnt)
                            elif blockIndex == 3:
                                print('Xception')
                                net=shufflenet_xception(net,inp, outp, base_mid_channels=outp // 2, stride=stride,
                                                                      activation=activation, useSE=useSE,scope_index=feature_cnt)
                            else:
                                raise NotImplementedError
                            input_channel = output_channel
                            feature_cnt+=1
                    fms.append(net)
                for item in fms:
                    print(item)

                if not include_head:
                    return fms

                if include_head:
                    x = slim.conv2d(net,
                                    num_outputs=1280,
                                    kernel_size=[1, 1],
                                    stride=1,
                                    activation_fn=hard_swish,
                                    normalizer_fn=slim.batch_norm,
                                    scope='conv_last/0')

                    x=tf.reduce_mean(x,axis=[1,2],keep_dims=True)

                    x=se(x,1280,scope='LastSE')



                    x = slim.conv2d(x,
                                    num_outputs=1280,
                                    kernel_size=[1, 1],
                                    stride=1,
                                    activation_fn=hard_swish,
                                    normalizer_fn=None,
                                    scope='fc/0')

                    x=slim.dropout(x,0.8,is_training=is_training)

                    x=slim.conv2d(x,
                                    num_outputs=cfg.MODEL.cls,
                                    kernel_size=[1, 1],
                                    stride=1,
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    scope='classifier/0')

        x=tf.squeeze(x, axis=1)
        x = tf.squeeze(x, axis=1)
        x=tf.identity(x,name='cls_output')
    return x




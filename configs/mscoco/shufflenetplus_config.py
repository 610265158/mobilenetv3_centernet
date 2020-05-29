#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu
config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.process_num = 3                      ### process_num for data provider
config.TRAIN.prefetch_size = 20                  ### prefect Q size for data provider

config.TRAIN.num_gpu = 1                         ##match with   os.environ["CUDA_VISIBLE_DEVICES"]
config.TRAIN.batch_size = 16                    ###A big batch size may achieve a better result, but the memory is a problem
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 300                      ###just keep training , evaluation shoule be take care by yourself,
                                               ### generally 10,0000 iters is enough

config.TRAIN.train_set_size=117266            ###widerface train size
config.TRAIN.val_set_size=5000             ###widerface val size

config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_value_every_step = [0.00001,0.0001,0.00025,0.000025,0.0000025,0.00000025]        ##warm up is used
config.TRAIN.lr_decay_every_step = [200,400,200000,300000,400000]

config.TRAIN.lr_decay='cos'
config.TRAIN.opt='adam'
config.TRAIN.weight_decay_factor = 1.e-4                  ##l2 regular
config.TRAIN.vis=False                                    ##check data flag
config.TRAIN.mix_precision=False

config.TRAIN.norm='BN'    ##'GN' OR 'BN'
config.TRAIN.lock_basenet_bn=False
config.TRAIN.frozen_stages=-1   ##no freeze
config.TRAIN.gradient_clip=False

config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.num_category=80                                  ###face 1  voc 20 coco 80
config.DATA.num_class = config.DATA.num_category

config.DATA.PIXEL_MEAN = [127.]                 ###rgb
config.DATA.PIXEL_STD = [127.]

config.DATA.hin = 520  # input size
config.DATA.win = 520
config.DATA.channel = 3
config.DATA.max_size=[config.DATA.hin,config.DATA.win]  ##h,w
config.DATA.cover_obj=4                          ###cover the small objs

config.DATA.mutiscale=False                #if muti scale set False  then config.DATA.MAX_SIZE will be the inputsize
config.DATA.scales=(320,640)
config.DATA.use_int8_data=True
config.DATA.use_int8_enlarge=255.           ### use uint8 for heatmap generate for less memory acc, to speed up
config.DATA.max_objs=128
config.DATA.cracy_crop=0.5
config.DATA.alpha=0.54
config.DATA.beta=0.54


##mobilenetv3 as basemodel
config.MODEL = edict()
config.MODEL.continue_train=False ### revover from a trained model
config.MODEL.model_path = './model/'  # save directory
config.MODEL.net_structure='ShuffleNetV2_Plus' ######'resnet_v1_50,resnet_v1_101,MobilenetV2
config.MODEL.size='Small'
config.MODEL.pretrained_model=None#'ShuffleNetV2+Small/ShuffleNetV2+Small.ckpt'
config.MODEL.task='mscoco'
config.MODEL.min_overlap=0.7
config.MODEL.max_box= 100
config.MODEL.offset= True
config.MODEL.global_stride=4
config.MODEL.head_dims=[64*3,64*3,32*3]

config.MODEL.deployee= False    ### tensorflow, mnn, coreml
if config.MODEL.deployee:
    config.TRAIN.batch_size = 1
    config.TRAIN.lock_basenet_bn=True




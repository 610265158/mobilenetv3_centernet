#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu
config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.process_num = 4                      ### process_num for data provider
config.TRAIN.prefetch_size = 20                  ### prefect Q size for data provider

config.TRAIN.num_gpu = 1                         ##match with   os.environ["CUDA_VISIBLE_DEVICES"]
config.TRAIN.batch_size = 16                    ###A big batch size may achieve a better result, but the memory is a problem
config.TRAIN.log_interval = 10
config.TRAIN.epoch = 300                        ###just keep training , evaluation shoule be take care by yourself,
                                                ### generally 10,0000 iters is enough

config.TRAIN.train_set_size=117266              ###coco train size
config.TRAIN.val_set_size=5000                  ###coco val size

config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.lr_value_every_step = [0.00001,0.0001,0.00025,0.0001,0.00001,0.000001]        ##warm up is used
config.TRAIN.lr_decay_every_step = [500,1000,150000,200000,250000]

config.TRAIN.opt='adam'
config.TRAIN.weight_decay_factor = 5.e-5                  ##l2 regular
config.TRAIN.vis=False                                    ##check data flag
config.TRAIN.mix_precision=True

config.TRAIN.norm='BN'    ##'GN' OR 'BN'
config.TRAIN.lock_basenet_bn=False
config.TRAIN.frozen_stages=-1   ##no freeze

config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.num_category=80                                  ###face 1  voc 20 coco 80
config.DATA.num_class = config.DATA.num_category         # +1 background

config.DATA.PIXEL_MEAN = [127.]                 ###rgb
config.DATA.PIXEL_STD = [127.]

config.DATA.hin = 416  # input size
config.DATA.win = 416
config.DATA.channel = 3
config.DATA.max_size=[config.DATA.hin,config.DATA.win]  ##h,w
config.DATA.cover_small_face=0                          ###cover the small faces

config.DATA.mutiscale=False                #if muti scale set False  then config.DATA.MAX_SIZE will be the inputsize
config.DATA.scales=(320,640)
config.DATA.use_int8_data=True
config.DATA.use_int8_enlarge=255.

# anchors -------------------------
config.ANCHOR = edict()
config.ANCHOR.rect=False
config.ANCHOR.rect_longer=False       ####    make anchor h/w=1.5
config.ANCHOR.ANCHOR_STRIDE = 16
config.ANCHOR.ANCHOR_SIZES = (32, 64, 128, 256, 320)   # sqrtarea of the anchor box
config.ANCHOR.ANCHOR_STRIDES = (8, 16, 32, 64, 128)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
config.ANCHOR.ANCHOR_RATIOS = (0.25, 1., 4.) ######           squrae
config.ANCHOR.ANCHOR_SCALES = (2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)) ######           1:2 in size,
config.ANCHOR.POSITIVE_ANCHOR_THRESH = 0.5
config.ANCHOR.NEGATIVE_ANCHOR_THRESH = 0.4

##mobilenetv3 as basemodel
config.MODEL = edict()
config.MODEL.continue_train=False ### revover from a trained model
config.MODEL.model_path = './model/'  # save directory
config.MODEL.net_structure='resnet_v2_50' ######'resnet_v1_50,resnet_v1_101,MobilenetV2
config.MODEL.pretrained_model='resnet_v2_50.ckpt'
config.MODEL.fpn_dims=[256,256,256,256,256]
config.MODEL.face=False
config.MODEL.min_overlap=0.7

config.MODEL.focal_loss=True
config.MODEL.fpn=True
config.MODEL.max_negatives_per_positive= 3.0


config.MODEL.deployee= False    ### tensorflow, mnn, coreml
if config.MODEL.deployee:
    config.TRAIN.batch_size = 1

config.MODEL.iou_thres= 0.05
config.MODEL.score_thres= 0.3
config.MODEL.max_box= 1500

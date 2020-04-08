#-*-coding:utf-8-*-

import os

from configs.shufflenetv2_config import config as shufflenet_config
from configs.mb_config import config as mb_config
from configs.mbv3_config import config as mb3_config
from configs.resnet_config import config as resnet_config
##### the config for different backbone,
config=mb3_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu
config.TRAIN.num_gpu = 1





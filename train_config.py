#-*-coding:utf-8-*-

import os

from configs.mscoco.mbv3_config import config as mb3_config
from configs.face.face_mbv3_config import config as face_mbv3_config
from configs.face.face_shufflenet_5x5_config import config as face_shufflenet_5x5_config
from configs.mscoco.shufflenetplus_config import config as shufflenet_plus_config
from configs.mscoco.shufflenet_5x5_config import config as shufflenet_5x5_config
##### the config for different task
config=mb3_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config.TRAIN.num_gpu = 1








import os
import random
import cv2
import numpy as np
import traceback

from lib.helper.logger import logger
from tensorpack.dataflow import DataFromGenerator
from tensorpack.dataflow import BatchData, PrefetchDataZMQ,RepeatedData


from lib.dataset.centernet_data_sampler import get_affine_transform,affine_transform
from lib.dataset.ttf_net_data_sampler import CenternetDatasampler


from lib.dataset.augmentor.augmentation import Random_scale_withbbox,\
                                                Random_flip,\
                                                baidu_aug,\
                                                dsfd_aug,\
                                                Fill_img,\
                                                Rotate_with_box,\
                                                produce_heatmaps_with_bbox,\
                                                box_in_img
from lib.dataset.augmentor.data_aug.bbox_util import *
from lib.dataset.augmentor.data_aug.data_aug import *
from lib.dataset.augmentor.visual_augmentation import ColorDistort,pixel_jitter

from lib.dataset.centernet_data_sampler import produce_heatmaps_with_bbox_official,affine_transform
from train_config import config as cfg


import math
import albumentations as A

class data_info():
    def __init__(self,img_root,txt):
        self.txt_file=txt
        self.root_path = img_root
        self.metas=[]


        self.read_txt()

    def read_txt(self):
        with open(self.txt_file) as _f:
            txt_lines=_f.readlines()
        txt_lines.sort()
        for line in txt_lines:
            line=line.rstrip()

            _img_path=line.rsplit('| ',1)[0]
            _label=line.rsplit('| ',1)[-1]

            current_img_path=os.path.join(self.root_path,_img_path)
            current_img_label=_label
            self.metas.append([current_img_path,current_img_label])

            ###some change can be made here
        logger.info('the dataset contains %d images'%(len(txt_lines)))
        logger.info('the datasets contains %d samples'%(len(self.metas)))


    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas

class MutiScaleBatcher(BatchData):

    def __init__(self, ds, batch_size,
                 remainder=False,
                 use_list=False,
                 scale_range=None,
                 input_size=(512,512),
                 divide_size=32,
                 is_training=True):
        """
        Args:
            ds (DataFlow): A dataflow that produces either list or dict.
                When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guaranteed to have the same batch size.
                If set to True, `len(ds)` must be accurate.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= len(ds)
            except NotImplementedError:
                pass

        self.batch_size = int(batch_size)
        self.remainder = remainder
        self.use_list = use_list

        self.scale_range=scale_range
        self.divide_size=divide_size

        self.input_size=input_size
        self.traing_flag=is_training



        self.target_producer=CenternetDatasampler()
    def __iter__(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """

        ##### pick a scale and shape aligment

        holder = []
        for data in self.ds:

            image,boxes_,klass_=data[0],data[1],data[2]




            data=[image,boxes_,klass_]
            holder.append(data)

            ### do crazy crop

            if random.uniform(0,1)<cfg.DATA.cracy_crop and self.traing_flag:
                if len(holder) == self.batch_size:
                    crazy_holder=[]
                    for i in range(0,len(holder),4):

                        crazy_iamge=np.zeros(shape=(2*cfg.DATA.hin,2*cfg.DATA.win,3),dtype=holder[i][0].dtype)

                        crazy_iamge[:cfg.DATA.hin,:cfg.DATA.win,:]=holder[i][0]
                        crazy_iamge[:cfg.DATA.hin, cfg.DATA.win:, :] = holder[i+1][0]
                        crazy_iamge[cfg.DATA.hin:, :cfg.DATA.win, :] = holder[i+2][0]
                        crazy_iamge[cfg.DATA.hin:, cfg.DATA.win:, :] = holder[i+3][0]



                        holder[i +1][1][:,[0, 2]]=holder[i +1][1][:,[0,2]]+cfg.DATA.win

                        holder[i + 2][1][:,[1, 3]] = holder[i + 2][1][:,[1, 3]] + cfg.DATA.hin

                        holder[i + 3][1][:,[0, 2]] = holder[i + 3][1][:,[0, 2]] + cfg.DATA.win
                        holder[i + 3][1][:,[1, 3]] = holder[i + 3][1][:,[1, 3]] + cfg.DATA.hin



                        tmp_bbox=np.concatenate((holder[i][1],
                                                holder[i+1][1],
                                                holder[i+2][1],
                                                holder[i+3][1]),
                                                axis=0)



                        tmp_klass = np.concatenate((holder[i][2] ,
                                                   holder[i + 1][2],
                                                   holder[i + 2][2],
                                                   holder[i + 3][2]),
                                                    axis=0)

                        ### do random crop 4 times:
                        for j in range(4):

                            curboxes=tmp_bbox.copy()
                            cur_klasses=tmp_klass.copy()
                            start_h=random.randint(0,cfg.DATA.hin)
                            start_w = random.randint(0, cfg.DATA.win)

                            cur_img_block=np.array(crazy_iamge[start_h:start_h+cfg.DATA.hin,start_w:start_w+cfg.DATA.win,:])

                            for k in range(len(curboxes)):
                                curboxes[k][0] = curboxes[k][0] - start_w
                                curboxes[k][1] = curboxes[k][1] - start_h
                                curboxes[k][2] = curboxes[k][2] - start_w
                                curboxes[k][3] = curboxes[k][3] - start_h

                            curboxes[:,[0, 2]] = np.clip(curboxes[:,[0, 2]], 0, cfg.DATA.win - 1)
                            curboxes[:,[1, 3]] = np.clip(curboxes[:,[1, 3]], 0, cfg.DATA.hin - 1)
                            ###cove the small faces




                            boxes_clean=[]
                            klsses_clean=[]
                            for k in range(curboxes.shape[0]):
                                box = curboxes[k]

                                if not ((box[3] - box[1]) < cfg.DATA.cover_obj or (
                                        box[2] - box[0]) < cfg.DATA.cover_obj):

                                    boxes_clean.append(curboxes[k])
                                    klsses_clean.append(cur_klasses[k])

                            boxes_clean=np.array(boxes_clean)
                            klsses_clean=np.array(klsses_clean)


                            crazy_holder.append([cur_img_block,boxes_clean,klsses_clean])

                    del holder

                    holder=crazy_holder


            if len(holder) == self.batch_size:
                target = self.produce_target(holder)

                yield BatchData.aggregate_batch(target, self.use_list)
                del holder[:]

        if self.remainder and len(holder) > 0:
            yield BatchData._aggregate_batch(holder, self.use_list)



    def produce_target(self,holder):
        alig_data = []

        if self.scale_range is not None:
            max_shape = [random.randint(*self.scale_range),random.randint(*self.scale_range)]

            max_shape[0] = int(np.ceil(max_shape[0] / self.divide_size) * self.divide_size)
            max_shape[1] = int(np.ceil(max_shape[1] / self.divide_size) * self.divide_size)

        else:
            max_shape=self.input_size

        # copy images to the upper left part of the image batch object
        for [image, boxes_, klass_] in holder:



            ### we do in map_function
            # image,boxes_=self.align_resize(image,boxes_,target_height=max_shape[0],target_width=max_shape[1])
            #
            # # construct an image batch object
            # image, shift_x, shift_y = self.place_image(image, target_height=max_shape[0], target_width=max_shape[1])
            # boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
            #image = image.astype(np.uint8)


            if cfg.TRAIN.vis:
                for __box in boxes_:

                    cv2.rectangle(image, (int(__box[0]), int(__box[1])),
                                  (int(__box[2]), int(__box[3])), (255, 0, 0), 4)

            heatmap, wh_map,weight = self.target_producer.ttfnet_centernet_datasampler(image,boxes_, klass_)

            if cfg.DATA.channel==1:
                image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
                image=np.expand_dims(image,-1)

            alig_data.append([image,heatmap, wh_map,weight])

        return alig_data



    def place_image(self,img_raw,target_height,target_width):



        channel = img_raw.shape[2]
        raw_height = img_raw.shape[0]
        raw_width = img_raw.shape[1]



        start_h=random.randint(0,target_height-raw_height)
        start_w=random.randint(0,target_width-raw_width)

        img_fill = np.zeros([target_height,target_width,channel], dtype=img_raw.dtype)
        img_fill[start_h:start_h+raw_height,start_w:start_w+raw_width]=img_raw

        return img_fill,start_w,start_h

    def align_resize(self,img_raw,boxes,target_height,target_width):
        ###sometimes use in objs detects
        h, w, c = img_raw.shape


        scale_y = target_height / h
        scale_x = target_width / w

        scale = min(scale_x, scale_y)

        image = cv2.resize(img_raw, None, fx=scale, fy=scale)
        boxes[:,:4]=boxes[:,:4]*scale

        return image, boxes


    def produce_for_centernet(self,image,boxes,klass,num_klass=cfg.DATA.num_class):
        # hm,reg_hm=produce_heatmaps_with_bbox(image,boxes,klass,num_klass)
        heatmap, wh,reg,ind,reg_mask = produce_heatmaps_with_bbox_official(image, boxes, klass, num_klass)
        return heatmap, wh,reg,ind,reg_mask


    def make_safe_box(self,image,boxes):
        h,w,c=image.shape

        boxes[boxes[:,0]<0]=0
        boxes[boxes[:, 1] < 0] = 0
        boxes[boxes[:, 2] >w] = w-1
        boxes[boxes[:, 3] >h] = h-1
        return boxes





class DsfdDataIter():

    def __init__(self, img_root_path='', ann_file=None, training_flag=True, shuffle=True):

        self.color_augmentor = ColorDistort()

        self.training_flag = training_flag

        self.lst = self.parse_file(img_root_path, ann_file)

        self.shuffle = shuffle

        self.train_trans = A.Compose([
                                      A.RandomBrightnessContrast(p=0.75, brightness_limit=0.1, contrast_limit=0.2),

                                      A.CLAHE(clip_limit=4.0, p=0.7),
                                      A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                                           val_shift_limit=10, p=0.5),

                                      A.OneOf([
                                          A.MotionBlur(blur_limit=5),
                                          A.MedianBlur(blur_limit=5),
                                          A.GaussianBlur(blur_limit=5),
                                          A.GaussNoise(var_limit=(5.0, 30.0)),
                                      ], p=0.7)
                                      ])

    def __iter__(self):
        idxs = np.arange(len(self.lst))

        while True:
            if self.shuffle:
                np.random.shuffle(idxs)
            for k in idxs:
                yield self._map_func(self.lst[k], self.training_flag)

    def __len__(self):
        return len(self.lst)

    def parse_file(self,im_root_path,ann_file):
        '''
        :return: [fname,lbel]     type:list
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file)
        all_samples = ann_info.get_all_sample()

        return all_samples

    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here
        try:
            fname, annos = dp
            image = cv2.imread(fname, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            labels = annos.split(' ')
            boxes = []


            for label in labels:
                bbox = np.array(label.split(','), dtype=np.float)
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])

            boxes = np.array(boxes, dtype=np.float)

            img=image

            if is_training:





                ###random crop and flip
                height, width = img.shape[0], img.shape[1]
                c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
                if 0:
                    input_h = (height | self.opt.pad) + 1
                    input_w = (width | self.opt.pad) + 1
                    s = np.array([input_w, input_h], dtype=np.float32)
                else:
                    s = max(img.shape[0], img.shape[1]) * 1.0
                    input_h, input_w = cfg.DATA.hin, cfg.DATA.win

                flipped = False
                if 1:
                    if 1:
                        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                        w_border = self._get_border(128, img.shape[1])
                        h_border = self._get_border(128, img.shape[0])
                        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

                    if np.random.random() < 0.5:
                        flipped = True
                        img = img[:, ::-1, :]
                        c[0] = width - c[0] - 1

                trans_output = get_affine_transform(c, s, 0, [input_w, input_h])

                inp = cv2.warpAffine(img, trans_output,
                                     (input_w, input_h),
                                     flags=cv2.INTER_LINEAR)

                boxes_ = boxes[:, :4]
                klass_ = boxes[:, 4:5]

                boxes_refine = []
                for k in range(boxes_.shape[0]):
                    bbox = boxes_[k]

                    cls_id = klass_[k]
                    if flipped:
                        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                    bbox[:2] = affine_transform(bbox[:2], trans_output)
                    bbox[2:] = affine_transform(bbox[2:], trans_output)
                    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, input_w - 1)
                    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, input_h - 1)

                    boxes_refine.append(bbox)

                boxes_refine = np.array(boxes_refine)
                image = inp.astype(np.uint8)

                # angle=random.choice([0,90,180,270])
                # image,boxes_refine=Rotate_with_box(image,angle,boxes_refine)

                boxes = np.concatenate([boxes_refine, klass_], axis=1)

                ####random crop and flip
                #### pixel level aug

                image=self.train_trans(image=image)['image']

                ####

            else:
                boxes_ = boxes[:, 0:4]
                klass_ = boxes[:, 4:]
                image, shift_x, shift_y = Fill_img(image, target_width=cfg.DATA.win, target_height=cfg.DATA.hin)
                boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
                h, w, _ = image.shape
                boxes_[:, 0] /= w
                boxes_[:, 1] /= h
                boxes_[:, 2] /= w
                boxes_[:, 3] /= h
                image = image.astype(np.uint8)
                image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

                boxes_[:, 0] *= cfg.DATA.win
                boxes_[:, 1] *= cfg.DATA.hin
                boxes_[:, 2] *= cfg.DATA.win
                boxes_[:, 3] *= cfg.DATA.hin
                image = image.astype(np.uint8)
                boxes = np.concatenate([boxes_, klass_], axis=1)

            if boxes.shape[0] == 0 or np.sum(image) == 0:
                boxes_ = np.array([[0, 0, -1, -1]])
                klass_ = np.array([0])
            else:
                boxes_ = np.array(boxes[:, 0:4], dtype=np.float32)
                klass_ = np.array(boxes[:, 4], dtype=np.int64)


        except:
            logger.warn('there is an err with %s' % fname)
            traceback.print_exc()
            image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.uint8)
            boxes_ = np.array([[0, 0, -1, -1]])
            klass_ = np.array([0])

        return image, boxes_, klass_

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i



class DataIter():
    def __init__(self, img_root_path='', ann_file=None, training_flag=True):

        self.shuffle = True
        self.training_flag = training_flag

        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size

        self.generator = DsfdDataIter(img_root_path, ann_file, self.training_flag )

        self.ds = self.build_iter()



    def parse_file(self, im_root_path, ann_file):

        raise NotImplementedError("you need implemented the parse func for your data")

    def build_iter(self,):


        ds = DataFromGenerator(self.generator)
        ds = RepeatedData(ds, -1)
        if cfg.DATA.mutiscale and self.training_flag:
            ds = MutiScaleBatcher(ds, self.num_gpu * self.batch_size,
                                  scale_range=cfg.DATA.scales,
                                  input_size=(cfg.DATA.hin, cfg.DATA.win),
                                  is_training=self.training_flag)
        else:
            ds = MutiScaleBatcher(ds, self.num_gpu * self.batch_size,
                                  input_size=(cfg.DATA.hin, cfg.DATA.win),
                                  is_training=self.training_flag)
        if not self.training_flag:
            self.process_num=1
        ds = PrefetchDataZMQ(ds, self.process_num, hwm=self.prefetch_size)
        ds.reset_state()
        ds = ds.get_data()
        return ds


    def __next__(self):
        return next(self.ds)


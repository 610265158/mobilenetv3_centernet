import sys
sys.path.append('.')
import numpy as np
from pycocotools.coco import COCO
import os

from lib.helper.logger import logger


## read coco data
class CocoMeta_keypoint:
    """ Be used in PoseInfo. """

    def __init__(self, idx, img_url, img_meta, keypoints, bbox):
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])
        self.bbox = bbox
        self.keypoints = []

        kp = keypoints

        #############reshape the keypoints for coco type,
        ################make the parts in legs unvisible

        #########################

        xs = kp[0::3]
        ys = kp[1::3]
        vs = kp[2::3]
        # if joint is marked
        joint_list = [(x, y, v) for x, y, v in zip(xs, ys, vs)]

        self.joint_list = []
        # 对原 COCO 数据集的转换 其中第二位之所以不一样是为了计算 Neck 等于左右 shoulder 的中点

        transform = list(
            zip([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
        prev_joint = joint_list

        new_joint = []
        for idx1, idx2 in transform:
            j1 = prev_joint[idx1 - 1]
            j2 = prev_joint[idx2 - 1]

            new_joint.append([(j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2, (j1[2] + j2[2]) / 2])

        # for background
        # new_joint.append([-1000, -1000])
        if len(new_joint) != 17 :
            print('The Length of joints list should be 0 or 17 but actually:', len(new_joint))
        self.keypoints = new_joint

class PoseInfo:
    """ Use COCO for pose estimation, returns images with people only. """

    def __init__(self, image_base_dir, anno_path):
        self.metas = []
        # self.data_dir = data_dir
        # self.data_type = data_type
        self.image_base_dir = image_base_dir
        self.anno_path = anno_path
        self.coco = COCO(self.anno_path)
        self.get_image_annos()
        self.image_list = os.listdir(self.image_base_dir)


    def get_image_annos(self):
        """Read JSON file, and get and check the image list.
        Skip missing images.
        """
        images_ids = self.coco.getImgIds()
        len_imgs = len(images_ids)
        for idx in range(len_imgs):

            images_info = self.coco.loadImgs([images_ids[idx]])
            image_path = os.path.join(self.image_base_dir, images_info[0]['file_name'])
            # filter that some images might not in the list
            if not os.path.exists(image_path):
                print("[skip] json annotation found, but cannot found image: {}".format(image_path))
                continue

            annos_ids = self.coco.getAnnIds(imgIds=[images_ids[idx]])
            annos_info = self.coco.loadAnns(annos_ids)

            anns = annos_info
            prev_center = []
            masks = []

            # sort from the biggest person to the smallest one
            if 1:

                persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

                for p_id in list(persons_ids):
                    p_counter = 0
                    ADD_FLAG = True
                    person_meta = anns[p_id]


                    # if person_meta["iscrowd"]:
                    #     continue
                    if person_meta["num_keypoints"] == 0:
                        continue
                    if person_meta["iscrowd"]==1:
                        continue
                    # skip this person if parts number is too low or if
                    # segmentation area is too small

                    person_box = [person_meta["bbox"][0], person_meta["bbox"][1], person_meta["bbox"][2],
                                  person_meta["bbox"][3]]
                    person_keypoints = person_meta["keypoints"]




                    ###some filter can be added here
                    # if person_box[2]* person_box[3]<32*32:
                    #     ADD_FLAG=False

                    # if person_keypoints[2]<=0:
                    #    p_counter+=1
                    ############如有手臂没标注就不要
                    # for i in range(15,33,3):
                    #    if person_keypoints[i+2]<=0:
                    #        p_counter+=1
                    # if p_counter>=7:
                    #    ADD_FLAG=False
                    ############################################################################

                    if ADD_FLAG:
                        meta = CocoMeta_keypoint(images_ids[idx], image_path, images_info[0], person_keypoints, person_box)
                        self.metas.append(meta)

        logger.info("Overall get {} valid pose images from {} and {}".format(
            len(self.metas), self.image_base_dir, self.anno_path))

    def load_images(self):
        pass

    def get_image_list(self):
        img_list = []
        for meta in self.metas:
            img_list.append(meta.img_url)
        return img_list

    def get_bbox(self):
        box_list = []
        for meta in self.metas:
            box_list.append(meta.bbox)
        return box_list

    def get_keypoints(self):
        keypoints_list = []
        for meta in self.metas:
            keypoints_list.append(meta.keypoints)
        return keypoints_list




## read coco data
class CocoMeta_bbox:
    """ Be used in PoseInfo. """


    ##this is the class_ map
    '''klass:(cat_id,'name),
         {0: (1, 'person'), 1: (2, 'bicycle'), 2: (3, 'car'), 3: (4, 'motorcycle'), 4: (5, 'airplane'), 5: (6, 'bus'),
         6: (7, 'train'), 7: (8, 'truck'), 8: (9, 'boat'), 9: (10, 'traffic light'), 10: (11, 'fire hydrant'),
         11: (13, 'stop sign'), 12: (14, 'parking meter'), 13: (15, 'bench'), 14: (16, 'bird'), 15: (17, 'cat'),
         16: (18, 'dog'), 17: (19, 'horse'), 18: (20, 'sheep'), 19: (21, 'cow'), 20: (22, 'elephant'),
         21: (23, 'bear'), 22: (24, 'zebra'), 23: (25, 'giraffe'), 24: (27, 'backpack'), 25: (28, 'umbrella'),
         26: (31, 'handbag'), 27: (32, 'tie'), 28: (33, 'suitcase'), 29: (34, 'frisbee'), 30: (35, 'skis'),
         31: (36, 'snowboard'), 32: (37, 'sports ball'), 33: (38, 'kite'), 34: (39, 'baseball bat'), 35: (40, 'baseball glove'),
         36: (41, 'skateboard'), 37: (42, 'surfboard'), 38: (43, 'tennis racket'), 39: (44, 'bottle'), 40: (46, 'wine glass'),
         41: (47, 'cup'), 42: (48, 'fork'), 43: (49, 'knife'), 44: (50, 'spoon'), 45: (51, 'bowl'),
         46: (52, 'banana'), 47: (53, 'apple'), 48: (54, 'sandwich'), 49: (55, 'orange'),   50: (56, 'broccoli'),
         51: (57, 'carrot'), 52: (58, 'hot dog'), 53: (59, 'pizza'), 54: (60, 'donut'), 55: (61, 'cake'),
         56: (62, 'chair'), 57: (63, 'couch'), 58: (64, 'potted plant'), 59: (65, 'bed'), 60: (67, 'dining table'),
         61: (70, 'toilet'), 62: (72, 'tv'), 63: (73, 'laptop'), 64: (74, 'mouse'), 65: (75, 'remote'),
         66: (76, 'keyboard'), 67: (77, 'cell phone'), 68: (78, 'microwave'), 69: (79, 'oven'), 70: (80, 'toaster'),
         71: (81, 'sink'), 72: (82, 'refrigerator'), 73: (84, 'book'), 74: (85, 'clock'), 75: (86, 'vase'),
         76: (87, 'scissors'), 77: (88, 'teddy bear'), 78: (89, 'hair drier'), 79: (90, 'toothbrush')}

        '''
    def __init__(self, idx, img_url, bbox):
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.bbox = bbox

        #############reshape the keypoints for coco type,
        ################make the parts in legs unvisible

        #########################


class BoxInfo:
    """ Use COCO for pose estimation, returns images with people only. """

    def __init__(self, image_base_dir, anno_path):
        self.metas = []
        # self.data_dir = data_dir
        # self.data_type = data_type
        self.image_base_dir = image_base_dir
        self.anno_path = anno_path
        self.coco = COCO(self.anno_path)
        self.get_image_annos()
        # self.image_list = os.listdir(self.image_base_dir)

    def get_image_annos(self):
        """Read JSON file, and get and check the image list.
        Skip missing images.
        """
        images_ids = self.coco.getImgIds()
        cats = self.coco.loadCats(self.coco.getCatIds())

        cat_klass_map={}

        for _cat in cats:
            cat_klass_map[_cat['id']]=_cat['name']

        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        print(cat_klass_map)

        len_imgs = len(images_ids)
        for idx in range(len_imgs):

            images_info = self.coco.loadImgs([images_ids[idx]])
            image_path = os.path.join(self.image_base_dir, images_info[0]['file_name'])
            # filter that some images might not in the list
            # if not os.path.exists(image_path):
            #     print("[skip] json annotation found, but cannot found image: {}".format(image_path))
            #     continue

            annos_ids = self.coco.getAnnIds(imgIds=[images_ids[idx]])
            annos_info = self.coco.loadAnns(annos_ids)



            bboxs=[]
            for ann in annos_info:

                if ann["iscrowd"]:
                    continue
                bbox = ann['bbox']
                cat = ann['category_id']
                klass = nms.index(cat_klass_map[cat])

                if bbox[2]<1 or bbox[3]<1:
                    continue

                bboxs.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], klass])

            if len(bboxs) > 0:
                tmp_meta = CocoMeta_bbox(images_ids[idx], image_path, bboxs)
                self.metas.append(tmp_meta)

            # sort from the biggest person to the smallest one

        logger.info("Overall get {} valid images from {} and {}".format(
            len(self.metas), self.image_base_dir, self.anno_path))

    def load_images(self):
        pass

    def get_image_list(self):
        img_list = []
        for meta in self.metas:
            img_list.append(meta.img_url)
        return img_list

    def get_bbox(self):
        box_list = []
        for meta in self.metas:
            box_list.append(meta.bbox)
        return box_list






if __name__=='__main__':



    #############bbox example
    coco_ann_path='/media/lz/023F3DA01938B2BB/mscoco/annotations/instances_val2017.json'
    coco_img_path='/media/lz/023F3DA01938B2BB/mscoco/val2017'
    coco_box=BoxInfo(coco_img_path,coco_ann_path)

    import cv2
    for meta in coco_box.metas:
        fname,bboxs=meta.img_url,meta.bbox

        image=cv2.imread(fname)
        image_show=image.copy()

        for bbox in bboxs:
            cv2.rectangle(image_show, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 0, 0), 7)

        cv2.imshow('tmp', image_show)
        cv2.waitKey(0)




    ###########keypoints example
import sys

sys.path.append('.')

import numpy as np

import math
import cv2
from train_config import config as cfg

def safe_box(bboxes,klasses):
    safe_box=[]
    safe_klass=[]
    for i in range(bboxes.shape[0]):
        cur_box=bboxes[i]
        cur_klass=klasses[i]
        x_min, y_min, x_max, y_max = cur_box[0], cur_box[1], cur_box[ 2], cur_box[ 3]

        if x_min<x_max  and y_min<y_max:
            safe_box.append(cur_box)
            safe_klass.append(cur_klass)


    return np.array(safe_box),np.array(safe_klass)

def bbox_areas(bboxes, keep_axis=False):

    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    areas = (y_max - y_min + 1) * (x_max - x_min + 1)

    if keep_axis:
        return areas[:, None]
    return areas


def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4)
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = np.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = np.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = np.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = np.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1] - 1)
        y1 = y1.clamp(min=0, max=featmap_size[0] - 1)
        x2 = x2.clamp(min=0, max=featmap_size[1] - 1)
        y2 = y2.clamp(min=0, max=featmap_size[0] - 1)
    return (x1, y1, x2, y2)


def torch_style_topK(matrix, K, axis=0):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(-matrix, axis=axis)

    ind=full_sort.take(np.arange(K), axis=axis)
    return matrix[ind],ind


class CenternetDatasampler:
    def __init__(self,
                 alpha=cfg.DATA.alpha,
                 beta=cfg.DATA.beta,
                 wh_agnostic=True,
                 wh_gaussian=True,
                 wh_area_process='log',
                 down_ratio=cfg.MODEL.global_stride):

        assert wh_area_process in [None, 'norm', 'log', 'sqrt']

        self.alpha=alpha
        self.beta=beta
        self.wh_area_process=wh_area_process

        self.down_ratio=down_ratio
        self.wh_agnostic = wh_agnostic,
        self.wh_gaussian = wh_gaussian,
        self.wh_planes=4


    def _ttfnet_gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        try:
            h[h < np.finfo(h.dtype).eps * h.max()] = 0

        except:
            print(h.shape)
        return h

    def draw_truncate_gaussian(self,heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = self._ttfnet_gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap
    def ttfnet_centernet_datasampler(self,image, gt_boxes, gt_labels, num_classes=cfg.DATA.num_class, max_objs=cfg.DATA.max_objs):

        """

        Args:
            gt_boxes: tensor, tensor <=> img, (num_gt, 4).
            gt_labels: tensor, tensor <=> img, (num_gt,).
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w) or (80 * 4, h, w).
            reg_weight: tensor, same as box_target
        """
        gt_boxes,gt_labels=safe_box(gt_boxes, gt_labels)


        img_h,img_w,_c=image.shape

        output_h, output_w = img_h//self.down_ratio,img_w//self.down_ratio


        heatmap_channel = num_classes

        heatmap = np.zeros((heatmap_channel, output_h, output_w),dtype=np.float32)
        fake_heatmap =np.zeros((output_h, output_w),dtype=np.float32)
        box_target = np.ones((self.wh_planes, output_h, output_w),dtype=np.float32) * -1
        reg_weight = np.zeros((self.wh_planes // 4, output_h, output_w),dtype=np.float32)


        if gt_boxes.shape[0]>0:

            if self.wh_area_process == 'log':
                boxes_areas_log = np.log(bbox_areas(gt_boxes))
            elif self.wh_area_process == 'sqrt':
                boxes_areas_log = np.sqrt(bbox_areas(gt_boxes))
            else:
                boxes_areas_log = bbox_areas(gt_boxes)

            boxes_area_topk_log, boxes_ind = torch_style_topK(boxes_areas_log, boxes_areas_log.shape[0])

            if self.wh_area_process == 'norm':
                boxes_area_topk_log[:] = 1.

            gt_boxes = gt_boxes[boxes_ind]
            gt_labels = gt_labels[boxes_ind]

            feat_gt_boxes = gt_boxes / self.down_ratio
            feat_gt_boxes[:, [0, 2]] = np.clip(feat_gt_boxes[:, [0, 2]], a_min=0,
                                                   a_max=output_w - 1)
            feat_gt_boxes[:, [1, 3]] = np.clip(feat_gt_boxes[:, [1, 3]], a_min=0,
                                                   a_max=output_h - 1)
            feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                                feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

            # we calc the center and ignore area based on the gt-boxes of the origin scale
            # no peak will fall between pixels
            ct_ints = (np.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                    (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                                   axis=1) / self.down_ratio).astype(np.int)


            h_radiuses_alpha = (feat_hs / 2. * self.alpha).astype(np.int)
            w_radiuses_alpha = (feat_ws / 2. * self.alpha).astype(np.int)

            if self.wh_gaussian and self.alpha != self.beta:
                h_radiuses_beta = (feat_hs / 2. * self.beta).astype(np.int)
                w_radiuses_beta = (feat_ws / 2. * self.beta).astype(np.int)

            if not self.wh_gaussian:
                # calculate positive (center) regions
                r1 = (1 - self.beta) / 2
                ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = calc_region(gt_boxes.transpose(0, 1), r1)
                ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s = [np.round(x.float() / self.down_ratio).int()
                                                      for x in [ctr_x1s, ctr_y1s, ctr_x2s, ctr_y2s]]
                ctr_x1s, ctr_x2s = [np.clamp(x, max=output_w - 1) for x in [ctr_x1s, ctr_x2s]]
                ctr_y1s, ctr_y2s = [np.clamp(y, max=output_h - 1) for y in [ctr_y1s, ctr_y2s]]
        else:
            boxes_ind=np.array([])
        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k]

            fake_heatmap = fake_heatmap*0

            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k], w_radiuses_alpha[k])

            heatmap[cls_id] = np.maximum(heatmap[cls_id], fake_heatmap)


            if self.wh_gaussian:
                if self.alpha != self.beta:
                    fake_heatmap = fake_heatmap*0
                    self.draw_truncate_gaussian(fake_heatmap,
                                                ct_ints[k],
                                                h_radiuses_beta[k],
                                                w_radiuses_beta[k])
                box_target_inds = fake_heatmap > 0
            else:
                ctr_x1, ctr_y1, ctr_x2, ctr_y2 = ctr_x1s[k], ctr_y1s[k], ctr_x2s[k], ctr_y2s[k]
                box_target_inds = np.zeros_like(fake_heatmap, dtype=np.uint8)
                box_target_inds[ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 + 1] = 1

            if self.wh_agnostic:

                box_target[:, box_target_inds] =np.expand_dims(gt_boxes[k],-1)

                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = np.expand_dims(gt_boxes[k],-1)

            if self.wh_gaussian:
                local_heatmap = fake_heatmap[box_target_inds]



                ct_div = local_heatmap.sum()
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div
            else:
                reg_weight[cls_id, box_target_inds] = \
                    boxes_area_topk_log[k] / box_target_inds.sum()


        heatmap = np.transpose(heatmap, axes=[1, 2, 0])
        box_target= np.transpose(box_target, axes=[1, 2, 0])
        reg_weight = np.transpose(reg_weight, axes=[1, 2, 0])





        if cfg.DATA.use_int8_data:

            heatmap = (heatmap * cfg.DATA.use_int8_enlarge).astype(np.uint8)
            return heatmap, box_target, reg_weight
        else:
            return heatmap, box_target, reg_weight





if __name__=='__main__':


    from train_config import config as cfg


    data_sampler=CenternetDatasampler()

    for i in range(1000):
        image = cv2.imread('./lib/dataset/augmentor/test.jpg')
        boxes = np.array([[165, 60, 233, 138],[5, 60, 133, 138]], dtype=np.float)

        cls=np.array([0,0])

        heatmap, box_target, reg_weight=data_sampler.ttfnet_centernet_datasampler(image,boxes,cls)

        hm=heatmap[:,:,0]
        wh = box_target[:, :, 1]+1

        weight=reg_weight[:, :, 0]

        print(np.max(wh))
        print(np.max(weight))
        cv2.namedWindow('image', 0)
        cv2.imshow('image', image)

        cv2.namedWindow('hm',0)
        cv2.imshow('hm',hm)

        cv2.namedWindow('weight', 0)
        cv2.imshow('weight', weight)

        cv2.namedWindow('wh', 0)
        cv2.imshow('wh', wh)
        cv2.waitKey(0)
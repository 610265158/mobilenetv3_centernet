#-*-coding:utf-8-*-
import numpy as np
import math
from train_config import config as cfg

def gaussian_radius(det_size, min_overlap=cfg.MODEL.min_overlap):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)


def draw_msra_gaussian(heatmap, center, sigma):
  #heatmap=np.transpose(heatmap,axes=[1,0])
  tmp_size = sigma * 3
  mu_x = int(center[0] + 0.5)
  mu_y = int(center[1] + 0.5)
  w, h = heatmap.shape[0], heatmap.shape[1]
  ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
  br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
  if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
    return heatmap
  size = 2 * tmp_size + 1
  x = np.arange(0, size, 1, np.float32)
  y = x[:, np.newaxis]
  x0 = y0 = size // 2
  g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
  g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
  g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
  img_x = max(0, ul[0]), min(br[0], h)
  img_y = max(0, ul[1]), min(br[1], w)
  heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
  #heatmap = np.transpose(heatmap, axes=[1, 0])
  return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def produce_heat_map(center, map_size, stride,objects_size, sigma,magic_divide=100):
    grid_y = map_size[0] // stride
    grid_x = map_size[1] // stride
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(grid_y)]
    x_range = [i for i in range(grid_x)]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start

    radis=gaussian_radius(objects_size)
    ratio=((objects_size[0]*objects_size[1]+0.000005)/(map_size[1]*map_size[0]))*magic_divide

    #d2 = (yy - center[0]) ** 2 / 2. / sigma_y / sigma_y + (xx - center[1]) ** 2 / 2. / sigma_x / sigma_x
    d2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma/ratio
    heatmap = np.exp(-exponent)

    am = np.amax(heatmap)
    if am > 0:
        heatmap /= am

    return heatmap

# def produce_heatmaps_with_bbox_official(image,boxes,klass,num_klass=cfg.DATA.num_class):
#     h_out, w_out, _ = image.shape
#     ## stride equal to 4
#     h_out //= 4
#     w_out //= 4
#     boxes[:, :4] //= 4
#
#     heatmap = np.zeros(shape=[h_out, w_out, num_klass],dtype=np.float32)
#
#     regression_map = np.zeros(shape=[h_out, w_out, 2],dtype=np.float32)
#
#     each_klass = set(klass)
#     for one_klass in each_klass:
#
#         for single_box, single_klass in zip(boxes, klass):
#             if single_klass == one_klass:
#                 ####box center (y,x)
#                 center = [round((single_box[1] + single_box[3]) / 2),
#                           round((single_box[0] + single_box[2]) / 2)]  ###0-1
#                 center = [int(x) for x in center]
#
#                 object_width = single_box[2] - single_box[0]
#                 object_height = single_box[3] - single_box[1]
#
#
#                 if center[0] >= h_out:
#                     center[0] -= 1
#                 if center[1] >= w_out:
#                     center[1] -= 1
#                 radius = gaussian_radius((math.ceil(object_height), math.ceil(object_width)))
#                 radius = max(0, int(radius))
#                 draw_msra_gaussian(heatmap[:, :, int(one_klass)],center,radius)
#
#                 regression_map[center[0], center[1], 0] = object_width
#                 regression_map[center[0], center[1], 1] = object_height
#
#
#     if cfg.DATA.use_int8_data:
#         h_am = np.amax(heatmap)
#
#         heatmap = (heatmap/h_am*cfg.DATA.use_int8_enlarge).astype(np.uint8)
#
#         regression_map=regression_map.astype(np.uint8)
#         return heatmap, regression_map
#     else:
#
#         return heatmap.astype(np.float16), regression_map.astype(np.float16)

def produce_heatmaps_with_bbox_official(image,boxes,klass,num_klass=cfg.DATA.num_class):
    return fuck(image,boxes,klass,num_klass)

def fuck(image,boxes,klass,num_classes=cfg.DATA.num_class,max_objs=128):
    h_out, w_out, _ = image.shape
    ## stride equal to 4
    output_h=h_out / 4
    output_w=w_out / 4
    boxes[:, :4] /= 4

    hm = np.zeros((num_classes, math.ceil(output_h), math.ceil(output_w)), dtype=np.float32)
    wh = np.zeros((max_objs, 2), dtype=np.float32)

    reg = np.zeros((max_objs, 2), dtype=np.float32)
    ind = np.zeros((max_objs), dtype=np.int64)
    reg_mask = np.zeros((max_objs), dtype=np.uint8)

    for k in range(len(boxes)):

        bbox = boxes[k]
        cls_id = klass[k]

        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0.00001, int(radius))

            ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_msra_gaussian(hm[cls_id], ct_int, radius)
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1

    heatmap=np.transpose(hm,axes=[1,2,0])

    if cfg.DATA.use_int8_data:

        heatmap = (heatmap*cfg.DATA.use_int8_enlarge).astype(np.uint8)

        return heatmap, wh,reg,ind,reg_mask
    else:
        return heatmap, wh,reg,ind,reg_mask


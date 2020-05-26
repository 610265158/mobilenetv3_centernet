#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np

def giou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0
    weight = tf.cast(weight[pos_mask],tf.float32)
    if avg_factor is None:
        avg_factor = tf.reduce_sum(pos_mask) + 1e-6
    bboxes1 = tf.reshape(pred[pos_mask],(-1, 4))
    bboxes2 =  tf.reshape(target[pos_mask],(-1, 4))


    lt = tf.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = tf.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = tf.maximum((rb - lt + 1),0)  # [rows, 2]
    enclose_x1y1 = tf.minimum(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = tf.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh =  tf.maximum((enclose_x2y2 - enclose_x1y1 + 1),0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
    u = ap + ag - overlap
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious
    return tf.reduce_sum(iou_distances * weight) / avg_factor

def diou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """DIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0
    weight = tf.cast(weight[pos_mask],tf.float32)
    if avg_factor is None:
        avg_factor = tf.reduce_sum(pos_mask) + 1e-6
    bboxes1 = tf.reshape(pred[pos_mask],(-1, 4))
    bboxes2 = tf.reshape(target[pos_mask],(-1, 4))


    lt = tf.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = tf.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = tf.maximum((rb - lt + 1),0)  # [rows, 2]
    # enclose_x1y1 = tf.minimum(bboxes1[:, :2], bboxes2[:, :2])
    # enclose_x2y2 = tf.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    # enclose_wh =  tf.maximum((enclose_x2y2 - enclose_x1y1 + 1),0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)


    # cal outer boxes
    outer_left_up = tf.minimum(bboxes1[:, :2], bboxes2[:, :2])
    outer_right_down = tf.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    outer = tf.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = tf.square(outer[:, 0]) + tf.square(outer[:, 1])

    boxes1_center = (bboxes1[:, :2] + bboxes1[:, 2:]+ 1) * 0.5
    boxes2_center = (bboxes2[:, :2] + bboxes2[:, 2:]+ 1) * 0.5
    center_dis = tf.square(boxes1_center[:, 0] - boxes2_center[:, 0]) + \
                 tf.square(boxes1_center[:, 1] - boxes2_center[:, 1])

    dious = ious - (center_dis / outer_diagonal_line)

    iou_distances = 1-dious

    return tf.reduce_sum(iou_distances * weight) / avg_factor
def ciou_loss(pred,
              target,
              weight,
              avg_factor=None):
    """GIoU loss.
    Computing the GIoU loss between a set of predicted bboxes and target bboxes.
    """
    pos_mask = weight > 0
    weight = tf.cast(weight[pos_mask],tf.float32)
    if avg_factor is None:
        avg_factor = tf.reduce_sum(pos_mask) + 1e-6
    bboxes1 = tf.reshape(pred[pos_mask],(-1, 4))
    bboxes2 = tf.reshape(target[pos_mask],(-1, 4))


    lt = tf.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = tf.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = tf.maximum((rb - lt + 1),0)  # [rows, 2]
    # enclose_x1y1 = tf.minimum(bboxes1[:, :2], bboxes2[:, :2])
    # enclose_x2y2 = tf.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    # enclose_wh =  tf.maximum((enclose_x2y2 - enclose_x1y1 + 1),0)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)




    # cal outer boxes
    outer_left_up = tf.minimum(bboxes1[:, :2], bboxes2[:, :2])
    outer_right_down = tf.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    outer = tf.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = tf.square(outer[:, 0]) + tf.square(outer[:, 1])


    boxes1_center = (bboxes1[:, :2] + bboxes1[:, 2:]+ 1) * 0.5
    boxes2_center = (bboxes2[:, :2] + bboxes2[:, 2:]+ 1) * 0.5
    center_dis = tf.square(boxes1_center[:, 0] - boxes2_center[:, 0]) + \
                 tf.square(boxes1_center[:, 1] - boxes2_center[:, 1])





    boxes1_size = tf.maximum(bboxes1[:,2:]-bboxes1[:,:2],0.0)
    boxes2_size = tf.maximum(bboxes2[:, 2:] - bboxes2[:, :2], 0.0)

    v = (4.0 / (np.pi**2)) * \
        tf.square(tf.math.atan(boxes2_size[:, 0] / (boxes2_size[:, 1]+0.00001)) -
                    tf.math.atan(boxes1_size[:, 0] / (boxes1_size[:, 1]+0.00001)))

    S = tf.cast(tf.greater(ious , 0.5),dtype=tf.float32)
    alpha = S * v / (1 - ious + v)

    cious = ious - (center_dis / outer_diagonal_line)-alpha * v

    cious = 1-cious

    return tf.reduce_sum(cious * weight) / avg_factor



if __name__=='__main__':
    gt=[[1000,10,100,100]]
    pre=[[200,200,1,1]]
    weight = [1]
    a = tf.constant(gt,dtype=tf.float32)
    b = tf.constant(pre,dtype=tf.float32)

    w=tf.constant(weight,dtype=tf.float32)

    session = tf.Session()

    loss,lt=giou_loss(pre,gt,w)
    v1 = session.run(loss)  # fetches参数为单个张量值，返回值为Numpy数组
    print(v1)
    lt = session.run(lt[0,:,:,0])  # fetches参数为单个张量值，返回值为Numpy数组
    print(lt.shape)
    print(lt)



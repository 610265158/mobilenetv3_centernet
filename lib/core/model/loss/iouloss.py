#-*-coding:utf-8-*-
import tensorflow as tf


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
        avg_factor = tf.reduce_sum(weight) + 1e-6
    bboxes1 = tf.reshape(pred,(-1, 4))
    bboxes2 = tf.reshape(target,(-1, 4))


    lt = tf.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = tf.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    wh = tf.clip_by_value((rb - lt + 1),clip_value_min=0,clip_value_max=99999)  # [rows, 2]
    enclose_x1y1 = tf.minimum(bboxes1[:, :2], bboxes2[:, :2])
    enclose_x2y2 = tf.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
    enclose_wh = tf.clip_by_value((enclose_x2y2 - enclose_x1y1 + 1),clip_value_min=0,clip_value_max=99999)

    overlap = wh[:, 0] * wh[:, 1]
    ap = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    ag = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (ap + ag - overlap)

    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]  # i.e. C in paper
    u = ap + ag - overlap
    gious = ious - (enclose_area - u) / enclose_area
    iou_distances = 1 - gious

    shifts_x = tf.range(0, (10 - 1) * 4 + 1, 4,
                       dtype=tf.int32)
    shifts_x = tf.cast(shifts_x, dtype=tf.float32)
    shifts_y = tf.range(0, (10 - 1) * 4 + 1, 4,
                        dtype=tf.int32)
    shifts_y = tf.cast(shifts_y, dtype=tf.float32)

    shift_y, shift_x = tf.meshgrid(shifts_y, shifts_x)

    base_loc = tf.stack((shift_y, shift_x), axis=2)  # (2, h, w)
    base_loc = tf.expand_dims(base_loc, axis=0)


    return tf.reduce_sum(iou_distances ),base_loc




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



#-*-coding:utf-8-*-

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops

from train_config import config as cfg

from lib.core.model.loss.iouloss import *

def loss(predicts,targets):
    pred_hm, pred_wh=predicts
    pred_hm=tf.nn.sigmoid(pred_hm)

    hm_target, wh_target,weights_=targets

    with tf.name_scope('losses'):
        # whether anchor is matched
        # shape [batch_size, num_anchors]

        with tf.name_scope('classification_loss'):
            hm_loss = focal_loss(
                pred_hm,
                hm_target
            )



        with tf.name_scope('iou_loss'):
            H, W = tf.shape(pred_hm)[1],tf.shape(pred_hm)[2]

            weights_=tf.transpose(weights_,perm=[0,3,1,2])
            mask = tf.reshape(weights_,shape=(-1, H, W))
            avg_factor = tf.reduce_sum(mask) + 1e-4

            base_step = cfg.MODEL.global_stride
            shifts_x = tf.range(0, (W - 1) * base_step + 1, base_step,
                                    dtype=tf.int32)
            shifts_x=tf.cast(shifts_x,dtype=tf.float32)
            shifts_y = tf.range(0, (H - 1) * base_step + 1, base_step,
                                    dtype=tf.int32)
            shifts_y = tf.cast(shifts_y, dtype=tf.float32)

            x_range, y_range = tf.meshgrid(shifts_x, shifts_y)

            base_loc = tf.stack((x_range, y_range), axis=2)  # (2, h, w)

            base_loc = tf.expand_dims(base_loc, axis=0)

            pred_boxes = tf.concat((base_loc[:,:,:,0:1] - pred_wh[:,:,:, 0:1],
                                    base_loc[:,:,:,1:2] - pred_wh[:,:,:, 1:2],
                                    base_loc[:,:,:,0:1] + pred_wh[:,:,:, 2:3],
                                    base_loc[:,:,:,1:2] + pred_wh[:,:,:, 3:4]), axis=3)

            # (batch, h, w, 4)
            boxes = wh_target#.permute(0, 2, 3, 1)

            wh_loss = ciou_loss(pred_boxes, boxes, mask, avg_factor=avg_factor)

        return hm_loss, wh_loss*5

def _reg_l1_loss(pred,
              target,
              weight,
              avg_factor=None):
    pos_mask = weight > 0
    weight = tf.cast(weight[pos_mask], tf.float32)
    if avg_factor is None:
        avg_factor = tf.reduce_sum(pos_mask) + 1e-6
    bboxes1 = tf.reshape(pred[pos_mask], (-1, 4))
    bboxes2 = tf.reshape(target[pos_mask], (-1, 4))


    loss=tf.reduce_mean(tf.abs(bboxes1-bboxes2),axis=1)
    return tf.reduce_sum(loss * weight) / avg_factor


def classification_loss(predictions, targets):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes + 1],
            representing the predicted logits for each class.
        targets: an int tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=predictions
    )
    return cross_entropy


def localization_loss(predictions, targets, indices, mask,sigma=9):
    """A usual L1 smooth loss.

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, 4],
            representing the (encoded) predicted locations of objects.
        targets: a float tensor with shape [batch_size, num_anchors, 4],
            representing the regression targets.
        weights: a float tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """

    indices = tf.where(tf.greater(targets, 0.))
    predictions = tf.gather_nd(predictions, indices)
    targets = tf.gather_nd(targets, indices)


    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0/sigma)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.maximum(1, tf.shape(indices)[0])
    normalizer = tf.cast(normalizer, dtype=tf.float32)

    return  tf.reduce_sum(tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5/sigma))/normalizer

def reg_l1_loss(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    k = tf.shape(indices)[1]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices, batch_dims=1)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss



# def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
#     r"""Compute focal loss for predictions.
#         Multi-labels Focal loss formula:
#             FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
#                  ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
#     Args:
#      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing the predicted logits for each class
#      target_tensor: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing one-hot encoded classification targets
#      weights: A float tensor of shape [batch_size, num_anchors]
#      alpha: A scalar tensor for focal loss alpha hyper-parameter
#      gamma: A scalar tensor for focal loss gamma hyper-parameter
#     Returns:
#         loss: A (scalar) tensor representing the value of the loss function
#     """
#
#
#     sigmoid_p = tf.nn.sigmoid(prediction_tensor)
#     zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
#
#     # For poitive prediction, only need consider front part loss, back part is 0;
#     # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
#     pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
#
#     # For negative prediction, only need consider back part loss, front part is 0;
#     # target_tensor > zeros <=> z=1, so negative coefficient = 0.
#     neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
#     per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
#                           - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
#
#
#     # compute the normalizer: the number of positive anchors
#     # normalizer = tf.where(tf.greater(target_tensor, 0))
#     # normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
#     # normalizer = tf.maximum(1., normalizer)
#
#
#     return tf.reduce_sum(per_entry_cross_ent)


def focal_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch,h,w,c)
        gt_regr (batch,h,w,c)
    '''
    pos_inds = tf.cast(tf.equal(gt, 1.0), dtype=tf.float32)
    neg_inds = 1.0 - pos_inds
    neg_weights = tf.pow(1.0 - gt, 4.0)

    pred = tf.clip_by_value(pred, 1e-6, 1.0 - 1e-6)
    pos_loss = tf.log(pred) * tf.pow(1.0 - pred, 2.0) * pos_inds
    neg_loss = tf.log(1.0 - pred) * tf.pow(pred, 2.0) * neg_weights * neg_inds

    num_pos = tf.reduce_sum(pos_inds)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    normalizer = tf.maximum(1., num_pos)
    loss = - (pos_loss + neg_loss) / normalizer

    return loss


def ohem_loss(logits, targets, weights):


    indices = tf.where(tf.not_equal(weights, -1))
    targets = tf.gather_nd(targets, indices)
    logits = tf.gather_nd(logits, indices)


    logits=tf.reshape(logits,shape=[-1,cfg.DATA.num_class])
    targets = tf.reshape(targets, shape=[-1])

    weights=tf.reshape(weights,shape=[-1])


    dtype = logits.dtype

    pmask = weights
    fpmask = tf.cast(pmask, dtype)
    n_positives = tf.reduce_sum(fpmask)


    no_classes = tf.cast(pmask, tf.int32)

    predictions = slim.softmax(logits)


    nmask = tf.logical_not(tf.cast(pmask,tf.bool))

    fnmask = tf.cast(nmask, dtype)

    nvalues = tf.where(nmask,
                       predictions[:, 0],
                       1. - fnmask)
    nvalues_flat = tf.reshape(nvalues, [-1])
    # Number of negative entries to select.
    max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
    n_neg = tf.cast(cfg.MODEL.max_negatives_per_positive * n_positives, tf.int32) + cfg.TRAIN.batch_size

    n_neg = tf.minimum(n_neg, max_neg_entries)

    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
    max_hard_pred = -val[-1]
    # Final negative mask.
    nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
    fnmask = tf.cast(nmask, dtype)

    # Add cross-entropy loss.
    with tf.name_scope('cross_entropy_pos'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=targets)

        neg_loss = tf.reduce_sum(loss * fpmask)

    with tf.name_scope('cross_entropy_neg'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=no_classes)
        pos_loss = tf.reduce_sum(loss * fnmask)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.where(tf.equal(weights, 1))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1., normalizer)

    return (neg_loss+pos_loss)/normalizer







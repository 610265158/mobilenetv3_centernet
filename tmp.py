# -*-coding:utf-8-*-
import tensorflow as tf

from lib.core.model.net.mobilenetv3.backbone import mobilenetv3_large_feature_extractor,\
                                                    mobilenetv3_large_minimalistic_feature_extractor
from lib.core.model.net.mobilenet.backbone import mobilenet_ssd

from lib.core.model.loss.centernet_loss import loss

from train_config import config as cfg

from lib.helper.logger import logger

from lib.core.model.head.centernet_head import CenternetHead

from lib.core.model.fpn.seperateconv_fpn import create_fpn_net
class Centernet():

    def __init__(self, ):

        if "MobilenetV2" in cfg.MODEL.net_structure:
            self.backbone = mobilenet_ssd
        elif "MobilenetV3" in cfg.MODEL.net_structure and not cfg.MODEL.minimalistic:
            self.backbone = mobilenetv3_large_feature_extractor
        elif "MobilenetV3" in cfg.MODEL.net_structure and cfg.MODEL.minimalistic:
            self.backbone = mobilenetv3_large_minimalistic_feature_extractor
        else:
            raise NotImplementedError

        self.head = CenternetHead()

        self.top_k_results_output = cfg.MODEL.max_box

    def forward(self, inputs, targets, training_flag):

        ## process the label
        if cfg.DATA.use_int8_data :
            inputs, targets = self.process_label(inputs, targets)

        ### extract feature maps
        origin_fms = self.backbone(inputs, training_flag)

        predictions = self.head(origin_fms, training_flag)

        ### calculate loss
        hm_loss, wh_loss = loss(predicts=predictions, targets=targets,base_step=cfg.MODEL.global_stride)

        self.postprocess(*predictions,self.top_k_results_output,cfg.MODEL.global_stride)


        return hm_loss, wh_loss

    def preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            image = image / 255.
        return image

    def process_label(self, inputs, targets):

        inputs = tf.cast(inputs, tf.float32)




        targets[0] = tf.cast(targets[0], tf.float32) / cfg.DATA.use_int8_enlarge

        return inputs, targets

    def postprocess(self, keypoints, wh, max_size,stride=4):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            where N = max_boxes.
        """

        def nms(heat, kernel=3):

            ##fast
            scores=tf.sigmoid(tf.reduce_max(heat,axis=3,keepdims=True))
            clses = tf.argmax(heat, axis=3)
            hmax = tf.layers.max_pooling2d(scores, kernel, 1, padding='same')
            keep = tf.cast(tf.equal(scores, hmax), tf.float32)

            return scores*keep , tf.cast(clses,tf.float32)

        def decode(heat, wh,stride, K=100):
            batch, H, W, cat = tf.shape(heat)[0], tf.shape(heat)[1], tf.shape(heat)[2], tf.shape(heat)[3]


            score_map, label_map = nms(heat)


            ### decode the box
            shifts_x = tf.range(0, (W - 1) * stride + 1, stride,
                                dtype=tf.int32)
            shifts_x = tf.cast(shifts_x, dtype=tf.float32)
            shifts_y = tf.range(0, (H - 1) * stride + 1, stride,
                                dtype=tf.int32)
            shifts_y = tf.cast(shifts_y, dtype=tf.float32)

            x_range, y_range = tf.meshgrid(shifts_x, shifts_y)

            base_loc = tf.stack((x_range, y_range), axis=2)  # (2, h, w)

            base_loc = tf.expand_dims(base_loc, axis=0)

            pred_boxes = base_loc-wh
            # pred_boxes = tf.concat((base_loc[:, :, :, 0:1] - wh[:, :, :, 0:1],
            #                         base_loc[:, :, :, 1:2] - wh[:, :, :, 1:2],
            #                         base_loc[:, :, :, 0:1] + wh[:, :, :, 2:3],
            #                         base_loc[:, :, :, 1:2] + wh[:, :, :, 3:4]), axis=3)



            ###get the topk bboxes
            score_map=tf.reshape(score_map,shape=[batch,-1])
            topk_scores, topk_inds = tf.nn.top_k(score_map, k=K)
            # # [b,k]

            pred_boxes=tf.reshape(pred_boxes,shape=[batch,-1,4])
            pred_boxes = tf.batch_gather(pred_boxes, topk_inds)

            label_map=tf.reshape(score_map,shape=[batch,-1])
            label_map = tf.batch_gather(label_map, topk_inds)

            topk_scores=tf.expand_dims(topk_scores,-1)
            label_map=tf.expand_dims(label_map,-1)
            detections=tf.concat([pred_boxes,topk_scores,label_map],axis=2)

            detections = tf.identity(detections, name='detections')

            return detections

        decode(keypoints, wh,stride, max_size)








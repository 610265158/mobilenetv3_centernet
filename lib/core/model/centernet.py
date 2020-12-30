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
        if cfg.DATA.use_int8_data:
            inputs, targets = self.process_label(inputs, targets)

        ### extract feature maps
        origin_fms = self.backbone(inputs, training_flag)

        fpn_fms=create_fpn_net(origin_fms)

        predictions = self.head(fpn_fms, training_flag)

        total_hm_loss=0
        total_wh_loss=0
        for i in range(3):

            ### calculate loss
            hm_loss, wh_loss = loss(predicts=predictions[i], targets=targets[i],base_step=2**(3+i))
            total_hm_loss+=hm_loss
            total_wh_loss+=wh_loss

        detections=[]

        for i in range(3):
            cur_res=self.postprocess(predictions[i][0], predictions[i][1],
                                     self.top_k_results_output,2**(3+i))

            detections.append(cur_res)

        result=tf.concat(detections,axis=1)

        result = tf.identity(result, name='detections')
        return total_hm_loss/3., total_wh_loss/3.

    def preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)

            image = image / 255.
        return image

    def process_label(self, inputs, targets):

        inputs = tf.cast(inputs, tf.float32)


        for i in range(3):

            targets[i][0] = tf.cast(targets[i][0], tf.float32) / cfg.DATA.use_int8_enlarge

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

            return scores*keep , clses

        def topk(hm, label, K=100):
            batch, height, width, cat = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
            # [b,h*w*c]
            scores = tf.reshape(hm, (batch, -1))
            labels = tf.reshape(label, (batch, -1))
            # [b,k]
            topk_scores, topk_inds = tf.nn.top_k(scores, k=K)
            # [b,k]
            topk_clses = tf.batch_gather(labels, topk_inds)

            topk_xs = topk_inds % width
            topk_ys = topk_inds // width

            return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

        def decode(heat, wh,stride, K=100):
            batch, height, width, cat = tf.shape(heat)[0], tf.shape(heat)[1], tf.shape(heat)[2], tf.shape(heat)[3]
            score_map, label_map = nms(heat)
            scores, inds, clses, ys, xs = topk(score_map, label_map, K=K)

            xs = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32)
            ys = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32)

            # [b,h*w,2]
            wh = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
            # [b,k,2]
            wh = tf.batch_gather(wh, inds)

            clses = tf.cast(tf.expand_dims(clses, axis=-1), tf.float32)
            scores = tf.expand_dims(scores, axis=-1)

            xmin = xs * stride - wh[:, :, 0:1]
            ymin = ys * stride - wh[:, :, 1:2]
            xmax = xs * stride + wh[:, :, 2:3]
            ymax = ys * stride + wh[:, :, 3:4]

            bboxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

            # [b,k,6]
            detections = tf.concat([bboxes, scores, clses], axis=-1)
            # detections = tf.identity(detections, name='detections')

            return detections

        res=decode(keypoints, wh,stride, max_size)

        return res







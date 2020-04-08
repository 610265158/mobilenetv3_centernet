import tensorflow as tf
import numpy as np
import cv2
import time
import math

from train_config import config as cfg



class FaceDetector:
    def __init__(self, model_path):
        """
        Arguments:
            model_path: a string, path to a pb file.
        """
        self._graph = tf.Graph()

        with self._graph.as_default():
            self._graph, self._sess = self.init_model(model_path)


            self.input_image = tf.get_default_graph().get_tensor_by_name('tower_0/images:0')
            self.training = tf.get_default_graph().get_tensor_by_name('training_flag:0')
            self.output_ops = [
                tf.get_default_graph().get_tensor_by_name('tower_0/boxes:0'),
                tf.get_default_graph().get_tensor_by_name('tower_0/scores:0'),
                tf.expand_dims(tf.cast(tf.get_default_graph().get_tensor_by_name('tower_0/labels:0'),dtype=tf.float32),-1)
            ]
            self.output_op=tf.concat(self.output_ops,axis=2)




    def __call__(self, image, score_threshold=0.5,input_shape=(cfg.DATA.hin,cfg.DATA.win),max_boxes=1000):
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 5].

        """


        # if input_shape is None:
        #     h, w, c = image.shape
        #     input_shape = (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)
        #
        # else:
        #     h, w = input_shape
        #     input_shape = (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)

        image, scale_x, scale_y, dx, dy = self.preprocess(image,
                                                                 target_height=cfg.DATA.hin,
                                                                 target_width=cfg.DATA.win)


        if cfg.DATA.channel==1:
            image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            image= np.expand_dims(image, -1)

        image_fornet = np.expand_dims(image, 0)


        bboxes = self._sess.run(
            self.output_op, feed_dict={self.input_image: image_fornet,self.training:False}
        )

        bboxes = self.py_nms(np.array(bboxes[0]), iou_thres=0.3, score_thres=score_threshold,max_boxes=max_boxes)

        ###recorver to raw image
        boxes_scaler = np.array([(input_shape[1]) / scale_x,
                                 (input_shape[0]) / scale_y,
                                 (input_shape[1]) / scale_x,
                                 (input_shape[0]) / scale_y,
                                 1.,1.], dtype='float32')

        boxes_bias = np.array([dx / scale_x,
                               dy / scale_y,
                               dx / scale_x,
                               dy / scale_y, 0.,0.], dtype='float32')
        bboxes = bboxes * boxes_scaler - boxes_bias



        # self.stats_graph(self._sess.graph)
        return bboxes


    def preprocess(self, image, target_height, target_width, label=None):

        ###sometimes use in objs detects
        h, w, c = image.shape

        bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype)

        scale_y = target_height / h
        scale_x = target_width / w

        scale = min(scale_x, scale_y)

        image = cv2.resize(image, None, fx=scale, fy=scale)

        h_, w_, _ = image.shape

        dx = (target_width - w_) // 2
        dy = (target_height - h_) // 2
        bimage[dy:h_ + dy, dx:w_ + dx, :] = image

        return bimage, scale, scale, dx, dy

    def py_nms(self, bboxes, iou_thres, score_thres, max_boxes=1000):

        upper_thres = np.where(bboxes[:, 4] > score_thres)[0]

        bboxes = bboxes[upper_thres]

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        order = np.argsort(bboxes[:, 4])[::-1]

        keep=[]
        while order.shape[0] > 0:
            if len(keep)>max_boxes:
                break
            cur = order[0]

            keep.append(cur)

            area = (bboxes[cur, 2] - bboxes[cur, 0]) * (bboxes[cur, 3] - bboxes[cur, 1])

            x1_reain = x1[order[1:]]
            y1_reain = y1[order[1:]]
            x2_reain = x2[order[1:]]
            y2_reain = y2[order[1:]]

            xx1 = np.maximum(bboxes[cur, 0], x1_reain)
            yy1 = np.maximum(bboxes[cur, 1], y1_reain)
            xx2 = np.minimum(bboxes[cur, 2], x2_reain)
            yy2 = np.minimum(bboxes[cur, 3], y2_reain)

            intersection = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)

            iou = intersection / (area + (y2_reain - y1_reain) * (x2_reain - x1_reain) - intersection)

            ##keep the low iou
            low_iou_position = np.where(iou < iou_thres)[0]

            order = order[low_iou_position + 1]

        return bboxes[keep]

    def stats_graph(self,graph):



        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        print(params)
        print('FLOPs: {}M;    Trainable params: {}'.format(flops.total_float_ops/1024/1024., params.total_parameters))

    def init_model(self,args):

        if len(args) == 1:
            use_pb = True
            pb_path = args[0]
        else:
            use_pb = False
            meta_path = args[0]
            restore_model_path = args[1]

        def ini_ckpt():
            graph = tf.Graph()
            graph.as_default()
            configProto = tf.ConfigProto()
            configProto.gpu_options.allow_growth = True
            sess = tf.Session(config=configProto)
            # load_model(model_path, sess)
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, restore_model_path)

            print("Model restred!")
            return (graph, sess)

        def init_pb(model_path):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            compute_graph = tf.Graph()
            compute_graph.as_default()
            sess = tf.Session(config=config)
            with tf.gfile.GFile(model_path, 'rb') as fid:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name='')


            # saver = tf.train.Saver(tf.global_variables())
            # saver.save(sess, save_path='./tmp.ckpt')
            return (compute_graph, sess)

        if use_pb:
            model = init_pb(pb_path)
        else:
            model = ini_ckpt()

        graph = model[0]
        sess = model[1]

        return graph, sess



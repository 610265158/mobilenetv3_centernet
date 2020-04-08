
import sys
sys.path.append('.')
from train_config import config as cfg
import tfcoreml
import coremltools
import cv2
import numpy as np


input_tensor = "tower_0/images:0"
bbox_output_tensor = "tower_0/boxes:0"
class_output_tensor = "tower_0/scores:0"


ssd_model=tfcoreml.convert(tf_model_path='./model/detector.pb',
                             mlmodel_path='./model/my_model.mlmodel',
                             image_input_names=input_tensor,
                             output_feature_names=[class_output_tensor,bbox_output_tensor],  # name of the output tensor (appended by ":0")
                             input_name_shape_dict={'tower_0/images:0': [1, cfg.DATA.hin, cfg.DATA.win, cfg.DATA.channel]},  # map from input tensor name (placeholder op in the graph) to shape
                             minimum_ios_deployment_target='12',
                             is_bgr=False)

spec = ssd_model.get_spec()
tfcoreml.optimize_nn_spec(spec)


#####clean the name of the model
print(spec.description)
spec.description.input[0].name = "image"
spec.description.input[0].shortDescription = "Input image"
spec.description.output[0].name = "scores"
spec.description.output[0].shortDescription = "Predicted class scores for each bounding box"
spec.description.output[1].name = "boxes"
spec.description.output[1].shortDescription = "Predicted coordinates for each bounding box"
#
#
input_mlmodel = input_tensor.replace(":", "__").replace("/", "__")
class_output_mlmodel = class_output_tensor.replace(":", "__").replace("/", "__")
bbox_output_mlmodel = bbox_output_tensor.replace(":", "__").replace("/", "__")

for i in range(len(spec.neuralNetwork.layers)):
    try:
        if spec.neuralNetwork.layers[i].input[0] == input_mlmodel:
            spec.neuralNetwork.layers[i].input[0] = "image"


        if class_output_mlmodel in spec.neuralNetwork.layers[i].output:
            spec.neuralNetwork.layers[i].output[1] = "scores"
        if spec.neuralNetwork.layers[i].output[0] == bbox_output_mlmodel:
            spec.neuralNetwork.layers[i].output[0] = "boxes"
    except:
        continue



#
num_classes = 1
num_anchors = 4200
spec.description.output[0].type.multiArrayType.shape.append(num_classes)
spec.description.output[0].type.multiArrayType.shape.append(num_anchors)
spec.description.output[0].type.multiArrayType.shape.append(1)

spec.description.output[1].type.multiArrayType.shape.append(4)
spec.description.output[1].type.multiArrayType.shape.append(num_anchors)
spec.description.output[1].type.multiArrayType.shape.append(1)
# print(spec.description)
#
# #
import coremltools
spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(spec)
ssd_model_re = coremltools.models.MLModel(spec)
#
# img=cv2.imread('./test2.jpg')
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img=cv2.resize(img,(640,640))
# img=np.array(img,dtype=np.float)
# img=np.expand_dims(img,axis=0)
#
# print(img.shape)
# coreml_inputs = {'image': img}
# coreml_output = ssd_model_re.predict(coreml_inputs, useCPUOnly=False)
#
# print(coreml_output['boxes'].shape)
# print(coreml_output['scores'].shape)



### transpose the axis

from coremltools.models import datatypes
from coremltools.models import neural_network

input_features = [
    ("scores", datatypes.Array(num_classes , num_anchors, 1)),
    ("boxes", datatypes.Array(4, num_anchors, 1))
]

output_features = [
    ("raw_confidence", datatypes.Array(num_anchors, num_classes)),
    ("raw_coordinates", datatypes.Array(num_anchors, 4))
]

builder = neural_network.NeuralNetworkBuilder(input_features, output_features)


builder.add_permute(name="permute_scores",
                    dim=(0, 3, 2, 1),
                    input_name="scores",
                    output_name="raw_confidence")

builder.add_permute(name="permute_output",
                    dim=(0, 3, 2, 1),
                    input_name="boxes",
                    output_name="raw_coordinates")
decoder_model = coremltools.models.MLModel(builder.spec)
# decoder_model.save("Decoder.mlmodel")


print(decoder_model.get_spec())

### add nms


nms_spec = coremltools.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3

for i in range(2):
    decoder_output = decoder_model._spec.description.output[i].SerializeToString()


    nms_spec.description.input.add()
    nms_spec.description.input[i].ParseFromString(decoder_output)

    nms_spec.description.output.add()
    nms_spec.description.output[i].ParseFromString(decoder_output)

nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"




num_classes=1
output_sizes = [num_classes, 4]
for i in range(2):
    ma_type = nms_spec.description.output[i].type.multiArrayType
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0
    ma_type.shapeRange.sizeRanges[0].upperBound = cfg.MODEL.max_box
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
    ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
    del ma_type.shape[:]

nms = nms_spec.nonMaximumSuppression
nms.confidenceInputFeatureName = "raw_confidence"
nms.coordinatesInputFeatureName = "raw_coordinates"
nms.confidenceOutputFeatureName = "confidence"
nms.coordinatesOutputFeatureName = "coordinates"
nms.iouThresholdInputFeatureName = "iouThreshold"
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

default_iou_threshold = 0.5
default_confidence_threshold = 0.0
nms.iouThreshold = default_iou_threshold
nms.confidenceThreshold = default_confidence_threshold

# nms.pickTop.perClass = False
# labels = np.loadtxt("coco_labels.txt", dtype=str, delimiter="\n")
# labels='qrcode'
# nms.stringClassLabels.vector.extend(labels)

nms_model = coremltools.models.MLModel(nms_spec)
# nms_model.save("NMS.mlmodel")


from coremltools.models import datatypes

from coremltools.models.pipeline import *

input_features = [ "image"]
output_features = [ "confidence", "coordinates" ]

pipeline = Pipeline(input_features, output_features)


pipeline.add_model(ssd_model_re)
pipeline.add_model(decoder_model)
pipeline.add_model(nms_model)

pipeline.spec.description.input[0].ParseFromString(
    ssd_model_re._spec.description.input[0].SerializeToString())
pipeline.spec.description.output[0].ParseFromString(
    nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(
    nms_model._spec.description.output[1].SerializeToString())

pipeline.spec.specificationVersion = 3


# pipeline.spec.input[0].type.multiArrayType.shape.append(num_classes)
final_model = coremltools.models.MLModel(pipeline.spec)
coreml_model_path='./coreml_mbv3_ssd.mlmodel'
final_model.save(coreml_model_path)


spec = final_model.get_spec()


print(spec.description)


img=cv2.imread('./test2.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img=cv2.resize(img,(640,640))

img_show=img
img=np.array(img,dtype=np.float)
img=np.expand_dims(img,axis=0)

print(img.shape)
coreml_inputs = {'image': img}
coreml_output = final_model.predict(coreml_inputs, useCPUOnly=False)

print(coreml_output['coordinates'].shape)
print(coreml_output['confidence'].shape)


score=coreml_output['confidence']
bbox=coreml_output['coordinates'][0]


cv2.rectangle(img_show, (int(bbox[0]*640), int(bbox[1]*640)),
                          (int(bbox[2]*640), int(bbox[3]*640)), (255, 0, 0), 4)
cv2.putText(img_show, str(score), (int(bbox[0]*640), int(bbox[1]*640) + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 0, 255), 2)

cv2.imshow('ss',img_show )
cv2.waitKey(0)
#
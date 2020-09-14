import coremltools as ct
import coremltools
from coremltools.models.neural_network import quantization_utils
from coremltools.models.neural_network.quantization_utils import AdvancedQuantizedLayerSelector

frozen_graph_file='./model/detector.pb'



fp_16_file='./centernet.mlmodel'



mlmodel = ct.convert(frozen_graph_file,inputs=[ct.ImageType()])

spec = mlmodel.get_spec()

print(mlmodel)

selector = AdvancedQuantizedLayerSelector(
    skip_layer_types=['batchnorm', 'depthwiseConv'],
    minimum_conv_kernel_channels=4,
    minimum_conv_weight_count=4096
)

model_fp16 = quantization_utils.quantize_weights(mlmodel, nbits=16,quantization_mode='linear',selector=selector)

model_fp16.save(fp_16_file)

print(model_fp16)

print('convert over, model was saved as ',fp_16_file)
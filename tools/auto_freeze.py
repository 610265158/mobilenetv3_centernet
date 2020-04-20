
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model", help="the trained file,  end with .ckpt",
                    type=str)
args = parser.parse_args()
pretrained_model=args.pretrained_model

print(pretrained_model)

command="python tools/centernet_for_freeze_bn.py --pretrained_model %s "%pretrained_model
os.system(command)
print('save ckpt with bn defaut False')




#### freeze again
model_folder = './model'
checkpoint = tf.train.get_checkpoint_state(model_folder)

##input_checkpoint
input_checkpoint = checkpoint.model_checkpoint_path
##input_graph
input_meta_graph = input_checkpoint + '.meta'

##output_node_names
output_node_names='tower_0/images,tower_0/detections'

#output_graph
output_graph='./model/detector.pb'

print('excuted')

command="python tools/freeze.py --input_checkpoint %s --input_meta_graph %s --output_node_names %s --output_graph %s"\
%(input_checkpoint,input_meta_graph,output_node_names,output_graph)
os.system(command)


print('detector.pb is saved with all feeeze')
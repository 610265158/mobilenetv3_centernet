 # mobilenetv3-ssd 检测算法的训练和部署
## 简介

light retinanet, end-to-end, including nms


### step by step

1. 下载代码，
2. 准备数据，mscoco
`python prepare_coco_data.py` 产生train.txt 和val.txt,
如果迁移其他任务，参考coco的方式组织数据集

3. 下载预训练模型，[mbv3](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_0.75_float.tgz)
4. python train.py
5. 完成训练后开始模型转换
6. 模型转换：这一块稍微复杂一点，主要是为了修改计算图，进行添加后处理，在转换pb的时候最大程度的自动优化。
6.1 . android，MNN
	6.1.1， 需要先修改configs/mbv3_config.py  下面的配置，
	```
	config.MODEL.continue_train=True
	config.MODEL.pretrained_model='yourmodel.ckpt'
	config.MODEL.deployee='mnn'
	其他后处理 比如iou_thres 等按需修改
	```
	6.1.2，运行一下 ` python train.py`, 加载训练好的参数，并立即保存
	6.1.3，`python tools/auto_freeze.py`, 产生pb模型
	6.1.4 然后用MNN 模型转换工具转换，具体怎么用不在赘述
6.2 ios, coreml
	6.2.1 需要先修改configs 下面的配置，
	```
	config.MODEL.continue_train=True
	config.MODEL.pretrained_model='yourmodel.ckpt'
	config.MODEL.deployee='coreml'
	其他后处理 比如iou_thres 等按需修改
	```
	6.2.2 运行一下 ` python train.py`, 加载训练好的参数，并立即保存
	6.2.3 `python tools/auto_freeze.py`, 产生pb模型
	6.2.4 `python tools/convert_to_coreml.py` 产生.mlmodel 模型, 放进xcode里就可以用了


然后再去部署就好了，整体上就是输入-输出， 输出格式是Nx4，Nx1，    即
[ [x1,y1,x2,y2],
[x3,y3,x4,y4],....]

[[score1,score2...]]
分别是坐标和得分，
此处的坐标都是相对值（0-1）。


### 
#
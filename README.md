# mobile_detect

## introduction

This is a tensorflow implement mobilenetv3-ssd framework,
which can be easily deployeed on both Android(MNN) and IOS(CoreML) mobile devices.

Purpose: Light detection algorithms that work on mobile devices is widely used, 
such as face detection, qrcode detection. 
So there is an easy project contains model training and model converter. 

** More convenient, the anchor decode  and the postprocess(nms) is included in the model. **
 
** contact me if u have question 2120140200@mail.nankai.edu.cn **



##pretrained model 
+ [baidu disk](https://pan.baidu.com/s/1FmALvtd8heKbus-sYzLr5A) ( password  rj94)
+ [google drive]()



##### (mnn  inference time 13ms including nms,inputs 480x480, on kirin980
##### coreml not tested)


## requirment

+ tensorflow1.14

+ tensorpack  (for data provider)

+ opencv

+ python 3.6

+ MNNConverter

+ tfcoreml

## useage

we make face detection as an example:

### train
1. download mscoco data

2. download pretrained model from 
[mbv3-large](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz)
[mbv3-large0.75](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_0.75_float.tgz)
[resnetv250](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)

3. then, run:

   ```python train.py```
   
   and if u want to check the data when training, u could set vis in train_config.py as True

4. After training, we should fix some params, such as training_flag, and batch=1, and so on, 
do as follow:

    4.1 android, mnn

    ```
    4.1.1 modify the configs/mbv3_cpnfig.py file 
    
        config.MODEL.continue_train=True
        config.MODEL.pretrained_model='yourmodel.ckpt'
        config.MODEL.deployee='mnn'
	
    4.1.2 python train.py, load and save the ckpt immediately,

    4.1.3 python tools/auto_freeze.py

    4.1.4 convert the model by MNNconverter, refer to MNN doc.

    ```
    
    4.2 ios, coreml

    ```
    4.2.1 modify the configs/mbv3_cpnfig.py file 
    
        config.MODEL.continue_train=True
        config.MODEL.pretrained_model='yourmodel.ckpt'
        config.MODEL.deployee='coreml'
    
    
    4.2.2 python train.py, load and save the ckpt immediately,

    4.2.3 python tools/auto_freeze.py
    4.2.4 python tools/convert_to_coreml.py produce coreml_mbv3.mlmodel
    ```


    
  Outputs of he models converted are nx4 and nx1, stand for coordinates and scores, and the coordinates are normalized


### evaluation

default scripts is for face detection.
** fddb **
```
    python model_eval/fddb.py [--model [TRAINED_MODEL]] [--data_dir [DATA_DIR]]
                          [--split_dir [SPLIT_DIR]] [--result [RESULT_DIR]]
    --model              Path of the saved model,default ./model/detector.pb
    --data_dir           Path of fddb all images
    --split_dir          Path of fddb folds
    --result             Path to save fddb results
 ```
    
example `python model_eval/fddb.py --model model/detector.pb 
                                    --data_dir 'fddb/img/' 
                                    --split_dir fddb/FDDB-folds/ 
                                    --result 'result/' `
                                    
** widerface **
```
    python model_eval/wider.py [--model [TRAINED_MODEL]] [--data_dir [DATA_DIR]]
                           [--result [RESULT_DIR]]
    --model              Path of the saved model,default ./model/detector.pb
    --data_dir           Path of WIDER
    --result             Path to save WIDERface results
 ```
example `python model_eval/wider.py --model model/detector.pb 
                                    --data_dir 'WIDER/WIDER_val/' 
                                    --result 'result/' `


### visualization

if u get a trained model and dont need to work on mobile device, run `python tools/auto_freeze.py`, it will read the checkpoint file in ./model, and produce detector.pb, then

`python vis.py`

u can check th code in vis.py to make it runable, it's simple.


### TODO: 
- [ ] Android project.

# mobile_centernet

## introduction

This is a tensorflow implement mobilenetv3-centernet framework,
which can be easily deployeed on both Android(MNN) and IOS(CoreML) mobile devices end to end.

Purpose: Light detection algorithms that work on mobile devices is widely used, 
such as face detection, targets detection.
So there is an easy project contains model training and model converter. 

** contact me if u have question 2120140200@mail.nankai.edu.cn **



## pretrained model , and preformance





## requirment

+ tensorflow 1.14

+ tensorpack 0.9.9  (for data provider)

+ opencv

+ python 3.6

+ MNNConverter

+ tfcoreml

## useage

### MSCOCO

#### train
1. download mscoco data

2. download pretrained model from
[mbv3-large0.75](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_0.75_float.tgz)


3. then, modify in config=mb3_config in train_config.py,  then run:

   ```python train.py```
   
   and if u want to check the data when training, u could set vis in train_config.py as True

4. After training, freeze the model as .pb  by

    ` python tools/auto_freeze.py --pretrained_mobile yourmodel.ckpt`

    it will produce a detector.pb

    4.1 MNN

    just use the converter, for example:
    `./MNNConvert -f TF --modelFile detector.pb --MNNModel centernet.mnn --bizCode biz  --fp16 1`

    4.2 coreml

    python tools/converter_to_coreml.py



#### evaluation

```
python model_eval/custome_eval.py [--model [TRAINED_MODEL]] [--annFile [cocostyle annFile]]
                          [--imgDir [the images dir]] [--is_show [show the result]]

python model_eval/custome_eval.py --model model/detector.pb
                                --annFile ../mscoco/annotations/instances_val2017.json
                                --imgDir ../mscoco/val2017
```

###  face

#### train

#### evaluation

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

`python visualization/vis.py`

u can check th code in visualization to make it runable, it's simple.


### TODO: 
- [ ] Android project.

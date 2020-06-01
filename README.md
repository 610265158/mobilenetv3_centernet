# mobile_centernet

## introduction

This is a tensorflow implement mobilenetv3-centernet framework,
which can be easily deployeed on both Android(MNN) and IOS(CoreML) mobile devices end to end.

Purpose: Light detection algorithms that work on mobile devices is widely used, 
such as face detection.
So there is an easy project contains model training and model converter. 

** contact me if u have question 2120140200@mail.nankai.edu.cn **



## pretrained model , and preformance

### mscoco

no test time augmentation,
| model                     |input_size |map      | map@0.5|map@0.75|
| :------:                  |:------:   |:------:  |:------:  |:------:  |
|[mbv3-large-0.75-modified_head](https://drive.google.com/open?id=1pUXyMEE3NPczfQ4jBcIFzzURIaqroVde)  |512x512     | 0.223    | 0.384|0.226  |


### fddb
| model                     |input_size |fddb      |
| :------:                  |:------:   |:------:  |
|  mbv3-small-minimalistic_modified  |512x512     | 0.947    |


### widerface
| model         |input_size  |wider easy|wider medium |wider hard |
| :------:      |:------:     |:------:  | :------:  | :------:  | 
|mbv3-small-minimalistic_modified |512x512      | 0.841    |0.803     |0.488    |


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
1. download mscoco data, then run `python prepare_coco_data.py --mscocodir ./mscoco`

2. download pretrained model from
[mbv3-large0.75](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_0.75_float.tgz)
relese it in the current dir.

3. then, modify in config=mb3_config in train_config.py,  then run:

   ```python train.py```
   
   and if u want to check the data when training, u could set vis in confifs/mscoco/mbv3_config.py as True

4. After training, freeze the model as .pb  by

    ` python tools/auto_freeze.py --pretrained_mobile ./model/yourmodel.ckpt`

    it will produce a detector.pb


#### evaluation

```
python model_eval/custome_eval.py [--model [TRAINED_MODEL]] [--annFile [cocostyle annFile]]
                          [--imgDir [the images dir]] [--is_show [show the result]]

python model_eval/custome_eval.py --model model/detector.pb
                                --annFile ../mscoco/annotations/instances_val2017.json
                                --imgDir ../mscoco/val2017
                                --is_show 1

ps, no test time augmentation is used.
```

###  FACE

#### train
1. download widerface data from http://shuoyang1213.me/WIDERFACE/,
 and release the WIDER_train, WIDER_val and wider_face_split into ./WIDER,
  then run python prepare_wider_data.py

2. download pretrained model from [mbv3-small-minimalistic](https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small-minimalistic_224_1.0_float.tgz)
release it in current dir

3. then, modify in config=face_config in train_config.py,  then run:

   ```python train.py```

   and if u want to check the data when training, u could set vis in confifs/mscoco/mbv3_config.py as True

4. After training, freeze the model as .pb  by

    ` python tools/auto_freeze.py --pretrained_mobile ./model/yourmodel.ckpt`

    it will produce a detector.pb


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

### finetune
1. download the trained model,
modify the config config.MODEL.pretrained_model='yourmodel.ckpt',
and set config.MODEL.continue_train=True
2. `python train.py`


### visualization

if u get a trained model and dont need to work on mobile device, run `python tools/auto_freeze.py`, it will read the checkpoint file in ./model, and produce detector.pb, then

`python visualization/vis.py`

u can check th code in visualization to make it runable, it's simple.


### model convert for mobile device
I have carefully processed the postprocess, and it can works within the model, so it could be deployed end to end.

4.1 MNN

    + 4.1.1 convert model

        just use the MNN converter, for example:
        `./MNNConvert -f TF --modelFile detector.pb --MNNModel centernet.mnn --bizCode biz  --fp16 1`

    + 4.1.2 visualization with mnn python wrapper

        `python visualization/vis_with_mnn.py --mnn_model centernet.mnn --imgDir 'your image dir'`

4.2 coreml

    + 4.2.1 convert

        `python tools/converter_to_coreml.py`

    + 4.2.2 visualization with coreml python wrapper

        `python visualization/vis_with_coreml.py --coreml_model centernet.mlmodel --imgDir 'your image dir'`

ps, if you want to do quantization, please reffer to the official doc, it is easy.

### TODO: 
- [ ] Android project.

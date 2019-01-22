# Inference


## Overview

Inference is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It 
allows user to conveniently use pre-trained models from Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
Inference provides multiple Scala interfaces.


### Highlights

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).

2. Support transformation of various input data type, thus supporting future prediction tasks.

3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed (up to 19.9x).

## Primary APIs

**InferenceModel**

`InferenceModel` is a thead-safe wrapper of AbstractModels, which can be used to load models and do the predictions.

***doLoad***

`doLoad` method is to load a bigdl, analytics-zoo model.

***doLoadCaffe***

`doLoadCaffe` method is to load a caffe model.

***doLoadTF***

`doLoadTF` method is to load a tensorflow model. The model can be loaded as a `FloatModel` or an `OpenVINOModel`. There are two backends to load a tensorflow model: TFNet and OpenVINO. 

<span id="jump">For OpenVINO backend, supported tensorflow models are listed below:</span>
                                          
    faster_rcnn_inception_resnet_v2_atrous_coco
    faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco
    faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid
    faster_rcnn_inception_resnet_v2_atrous_oid
    faster_rcnn_nas_coco
    faster_rcnn_nas_lowproposals_coco
    faster_rcnn_resnet101_coco
    faster_rcnn_resnet101_kitti
    faster_rcnn_resnet101_lowproposals_coco
    mask_rcnn_inception_resnet_v2_atrous_coco
    mask_rcnn_inception_v2_coco
    mask_rcnn_resnet101_atrous_coco
    mask_rcnn_resnet50_atrous_coco
    ssd_inception_v2_coco
    ssd_mobilenet_v1_coco
    ssd_mobilenet_v2_coco
    ssdlite_mobilenet_v2_coco
                                          
***doLoadOpenVINO***
                                          
`doLoadOpenVINO` method is to load an OpenVINO Intermediate Representation(IR).

***doReload***

`doReload` method is to reload the bigdl, analytics-zoo model.

***doPredict***

`doPredict` method is to do the prediction.

**InferenceSupportive**

`InferenceSupportive` is a trait containing several methods for type transformation, which transfer a model input 
to a valid data type, thus supporting future inference model prediction tasks.

For example, method `transferTensorToJTensor` convert a model input of data type `Tensor` 
to [`JTensor`](https://github.com/intel-analytics/analytics-zoo/blob/88afc2d921bb50341d8d7e02d380fa28f49d246b/zoo/src/main/java/com/intel/analytics/zoo/pipeline/inference/JTensor.java)
, which will be the input for a FloatInferenceModel.

**AbstractModel**

`AbstractModel` is an abstract class to provide APIs for basic functions - `predict` interface for prediction, `copy` interface for coping the model into the queue of AbstractModels, `release` interface for releasing the model and `isReleased` interface for checking the state of model release.  

**FloatModel**

`FloatModel` is an extending class of `AbstractModel` and achieves all `AbstractModel` interfaces.

**OpenVINOModel**

`OpenVINOModel` is an extending class of `AbstractModel`. It achieves all `AbstractModel` functions.

**InferenceModelFactory**

`InferenceModelFactory` is an object with APIs for loading pre-trained Analytics Zoo models, Caffe models, Tensorflow models and OpenVINO Intermediate Representations(IR).
Analytics Zoo models, Caffe models, Tensorflow models can be loaded as FloatModels. The load result of it is a `FloatModel`
Tensorflow models and OpenVINO Intermediate Representations(IR) can be loaded as OpenVINOModels. The load result of it is an `OpenVINOModel`. 
The load result of it is a `FloatModel` or an `OpenVINOModel`. 


**OpenVinoInferenceSupportive**

`OpenVinoInferenceSupportive` is an extending object of `InferenceSupportive` and focus on the implementation of loading pre-trained models, including tensorflow models and OpenVINO Intermediate Representations(IR). 
There are two backends to load a tensorflow model: TFNet and OpenVINO. For OpenVINO backend, [supported tensorflow models](#jump) are listed in the section of `doLoadTF` method of `InferenceModel` API above. 





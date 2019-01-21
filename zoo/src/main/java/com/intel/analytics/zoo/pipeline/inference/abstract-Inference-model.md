# Abstract Inference Model

## Overview

Abstract inference model is an abstract class in Analytics Zoo aiming to provide support for 
java implementation in loading a collection of pre-trained models(including Caffe models, 
Tensorflow models, OpenVINO Intermediate Representations(IR), etc.) and for model prediction.
AbstractInferenceModel contains a mix of methods declared with implementation for loading models and prediction.
You will need to create a subclass which extends the AbstractInferenceModel to 
develop your java applications.

### Highlights

1. Easy-to-use java API for loading and prediction with deep learning models.

2. In a few lines, run large scale inference from pre-trained models of Analytics-Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).

3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed (up to 19.9x).

## Primary APIs

**load**

AbstractInferenceModel provides `load` API for loading a pre-trained model,
thus we can conveniently load various kinds of pre-trained models in java applications. The load result of
`AbstractInferenceModel` is an `AbstractModel`.
We just need to specify the model path and optionally weight path if exists where we previously saved the model.

***load***

`load` method is to load a BigDL model.

***loadCaffe***

`loadCaffe` method is to load a caffe model.

***loadTF***

`loadTF` method is to load a tensorflow model. There are two backends to load a tensorflow model and to do the predictions: TFNet and OpenVINO. For OpenVINO backend, supported tensorflow models are listed below:

    embedded_ssd_mobilenet_v1_coco
    facessd_mobilenet_v2_quantized_320x320_open_image_v4
    faster_rcnn_inception_resnet_v2_atrous_coco
    faster_rcnn_inception_resnet_v2_atrous_cosine_lr_coco
    faster_rcnn_inception_resnet_v2_atrous_oid
    faster_rcnn_inception_resnet_v2_atrous_pets
    faster_rcnn_inception_v2_coco
    faster_rcnn_inception_v2_pets
    faster_rcnn_nas_coco
    faster_rcnn_resnet101_atrous_coco
    faster_rcnn_resnet101_ava_v2.1
    faster_rcnn_resnet101_coco
    faster_rcnn_resnet101_fgvc
    faster_rcnn_resnet101_kitti
    faster_rcnn_resnet101_pets
    faster_rcnn_resnet101_voc07
    faster_rcnn_resnet152_coco
    faster_rcnn_resnet152_pets
    faster_rcnn_resnet50_coco
    faster_rcnn_resnet50_fgvc
    faster_rcnn_resnet50_pets
    mask_rcnn_inception_resnet_v2_atrous_coco
    mask_rcnn_inception_v2_coco
    mask_rcnn_resnet101_atrous_coco
    mask_rcnn_resnet101_pets
    mask_rcnn_resnet50_atrous_coco
    rfcn_resnet101_coco
    rfcn_resnet101_pets
    ssd_inception_v2_coco
    ssd_inception_v2_pets
    ssd_inception_v3_pets
    ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync
    ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync
    ssd_mobilenet_v1_0.75_depth_quantized_300x300_pets_sync
    ssd_mobilenet_v1_300x300_coco14_sync
    ssd_mobilenet_v1_coco
    ssd_mobilenet_v1_focal_loss_pets
    ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync
    ssd_mobilenet_v1_pets
    ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync
    ssd_mobilenet_v1_quantized_300x300_coco14_sync
    ssd_mobilenet_v2_coco
    ssd_mobilenet_v2_quantized_300x300_coco
    ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync
    ssdlite_mobilenet_v1_coco
    ssdlite_mobilenet_v2_coco

***loadOpenVINO***

`loadOpenVINO` method is to load an OpenVINO Intermediate Representation(IR).

**predict**

AbstractInferenceModel provides `predict` API for prediction with loaded model.
The predict result of`AbstractInferenceModel` is a `List<List<JTensor>>` by default.

## Examples

It's very easy to apply abstract inference model for inference with below code piece.
You will need to write a subclass that extends AbstractinferenceModel.
```java
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class TextClassificationModel extends AbstractInferenceModel {
    public TextClassificationModel() {
        super();
    }
 }
TextClassificationModel model = new TextClassificationModel();
model.load(modelPath, weightPath);
List<List<JTensor>> result = model.predict(inputList);
```

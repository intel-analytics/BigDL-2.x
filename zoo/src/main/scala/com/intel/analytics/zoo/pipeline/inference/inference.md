# Inference


## Overview

Inference is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It 
allows user to conveniently use pre-trained models from Analytics Zoo, Tensorflow and Caffe.
Inference provides multiple Scala interfaces.


### Highlights

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow.

2. Support transformation of various input data type, thus supporting future prediction tasks.

## Primary APIs


**InferenceSupportive**

`InferenceSupportive` is a trait containing several methods for type transformation, which transfer a model input 
to a valid data type, thus supporting future inference model prediction tasks.

For example, method `transferTensorToJTensor` convert a model input of data type `Tensor` 
to [`JTensor`](https://github.com/intel-analytics/analytics-zoo/blob/88afc2d921bb50341d8d7e02d380fa28f49d246b/zoo/src/main/java/com/intel/analytics/zoo/pipeline/inference/JTensor.java)
, which will be the input for a FloatInferenceModel.

**FloatInferenceModel**

`FloatInferenceModel` is an extending class of `InferenceSupportive` and additionally provides `predict` API for prediction tasks.

**InferenceModelFactory**

`InferenceModelFactory` is an object with APIs for loading pre-trained Analytics Zoo models, Caffe models and Tensorflow models.
We just need to specify the model path and optionally weight path if exists where we previously saved the model.
The load result of is a `FloatInferenceModel`.


**ModelLoader**

`ModelLoader` is an extending object of  `InferenceSupportive` and focus on the implementation of loading pre-trained models.






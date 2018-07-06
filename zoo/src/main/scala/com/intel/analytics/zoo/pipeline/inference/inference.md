# Inference


## Overview

Inference is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It 
allows user to combine the power of Analytics Zoo, Tensorflow and Caffe.
Inference provides multiple Scala interfaces.


**Highlights**

1. Easy-to-use DataFrame(DataSet)-based API for training, prediction and evaluation with deep learning models.

2. Effortless integration with Spark ML pipeline and compatibility with other feature transformers and algorithms in Spark ML.

3. In a few lines, run large scale inference or transfer learning from pre-trained models of Caffe, Keras, Tensorflow or BigDL.

4. Training of customized model or BigDL built-in neural models (e.g. Inception, ResNet, Wide And Deep).

5. Rich toolset for feature extraction and processing, including image, audio and texts.



## Primary APIs

**FloatInferenceModel**

`FloatInferenceModel` is a class whiich extends `InferenceSupportive` and provides `predict` API for prediction tasks.

**InferenceModelFactory**

`InferenceModelFactory` provides high level API for for loading pre-trained Analytics Zoo models, Caffe models and Tensorflow models.
We just need to specify the model path and optionally weight path if exists where we previously saved the model.


**Inference Supportive**

`NNClassifier` and `NNClassifierModel`extends `NNEstimator` and `NNModel` and focus on 
classification tasks, where both label column and prediction column are of Double type.

**ModelLoader**
`ModelLoader` extends inference Supportive and focus on model loading tasks.


please check our
[ImageProcessing](../APIGuide/PipelineAPI/nnframes.md#NNImageReader) for detailed usage.

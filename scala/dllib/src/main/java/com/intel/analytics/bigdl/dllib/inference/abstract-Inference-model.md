# Abstract Inference Model

## Overview

Abstract inference model is an abstract class in Analytics Zoo aiming to provide support for 
java implementation in loading a collection of pre-trained models(including Caffe models, 
Tensorflow models, etc.) and for model prediction.
AbstractInferenceModel contains a mix of methods declared with implementation for loading models and prediction.
You will need to create a subclass which extends the AbstractInferenceModel to 
develop your java applications.

### Highlights

1. Easy-to-use java API for loading and prediction with deep learning models.

2. In a few lines, run large scale inference from pre-trained models of Analytics-Zoo, Caffe, Tensorflow.


## Primary APIs

**load**

AbstractInferenceModel provides `load` API for loading a pre-trained model,
thus we can conveniently load various kinds of pre-trained models in java applications. The load result of
`AbstractInferenceModel` is a [`FloatInferenceModel`](https://github.com/xuex2017/analytics-zoo/blob/88afc2d921bb50341d8d7e02d380fa28f49d246b/zoo/src/main/scala/com/intel/analytics/zoo/pipeline/inference/FloatInferenceModel.scala). 
We just need to specify the model path and optionally weight path if exists where we previously saved the model.

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

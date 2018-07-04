# Abstract Inference Model

## Overview

Abstract inference model is an abstract class in Analytics Zoo aiming to provide support for loading a collection of 
pre-trained models(including Caffe models, Tensorflow models, etc.) and for model prediction.

### Highlights

## Examples

It's very easy to apply abstract inference model for inference with below code piece.
You will need to write a subclass and extends AbstractinferenceModel.
```java
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class TextClassificationModel extends AbstractInferenceModel {
    public TextClassificationModel() {
        super();
    }
 }
TextClassificationModel model = new TextClassificationModel();
model.load(modelPath);
List<List<JTensor>> result = model.predict(inputList);
```

## Primary APIs



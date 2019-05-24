Inference Model is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It allows user to conveniently use pre-trained models from Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). Inference Model provides Java, Scala and Python interfaces.

**Highlights**

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
2. Support transformation of various input data type, thus supporting future prediction tasks.
3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).

**Basic usage of Inference Model:**

1. Directly use InferenceModel or write a subclass extends `InferenceModel` (`AbstractinferenceModel` in Java).
2. Load pre-trained models with corresponding `load` methods, e.g, doLoad for Zoo, and doLoadTF for TensorFlow.
3. Do prediction with `predict` method.

Note that OpenVINO required extra [dependencies](https://github.com/opencv/dldt/blob/2018_R5/inference-engine/install_dependencies.sh).

**Supported models:**

1. [Zoo Models](https://analytics-zoo.github.io/master/##built-in-deep-learning-models)
2. [Caffe Models](https://github.com/BVLC/caffe/wiki/Model-Zoo)
3. [TensorFlow Models](https://github.com/tensorflow/models)
4. [OpenVINO models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)

## Load pre-trained model
### Load pre-trained Zoo/bigDL model

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.load(modelPath, weightPath);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoad(modelPath, weightPath)
```

**Python**

```python
model = InferenceModel()
model.load(modelPath, weightPath)
```

* `modelPath`: String. Path of pre-trained model.
* `weightPath`: String. Path of pre-trained model weight. Default is `null`.

### Load pre-trained Caffe model

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.loadCaffe(modelPath, weightPath);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoadCaffe(modelPath, weightPath)
```

**Python**

```python
model = InferenceModel()
model.load_caffe(modelPath, weightPath)
```

* `modelPath`: String. Path of pre-trained model.
* `weightPath`: String. Path of pre-trained model weight.

### Load TensorFlow model

1. **Load with TensorFlow backend**

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.loadCaffe(modelPath, weightPath);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoadCaffe(modelPath, weightPath)
```

**Python**

```python
model = InferenceModel()
model.load_caffe(modelPath, weightPath)
```

* `modelPath`: String. Path of pre-trained model.
* `weightPath`: String. Path of pre-trained model weight.

2. **Load with OpenVINO backend**

**Java**

```java
public class ExtendedInferenceModel extends AbstractInferenceModel {
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
model.loadCaffe(modelPath, weightPath);
```

**Scala**

```scala
val model = new InferenceModel()
model.doLoadCaffe(modelPath, weightPath)
```

**Python**

```python
model = InferenceModel()
model.load_caffe(modelPath, weightPath)
```

* `modelPath`: String. Path of pre-trained model.
* `weightPath`: String. Path of pre-trained model weight.

### Load OpenVINO model

## Predict
Do prediction with `predict` method. Input


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

`loadTF` method is to load a tensorflow model. There are two backends to load a tensorflow model and to do the predictions: TFNet and OpenVINO. For OpenVINO backend, [supported tensorflow models](https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) are listed below:

***loadOpenVINO***

`loadOpenVINO` method is to load an OpenVINO Intermediate Representation(IR).

***loadOpenVINOInt8***

`loadOpenVINO` method is to load an OpenVINO Int8 Intermediate Representation(IR).

**predict**

AbstractInferenceModel provides `predict` API for prediction with loaded model.
The predict result of`AbstractInferenceModel` is a `List<List<JTensor>>` by default.

**predictInt8**

AbstractInferenceModel provides `predictInt8` API for prediction with loaded int8 model.
The predictInt8 result of`AbstractInferenceModel` is a `List<List<JTensor>>` by default.

### JAVA Examples

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

## Primary APIs for Scala

**InferenceModel**

`InferenceModel` is a thead-safe wrapper of AbstractModels, which can be used to load models and do the predictions.

***doLoad***

`doLoad` method is to load a bigdl, analytics-zoo model.

***doLoadCaffe***

`doLoadCaffe` method is to load a caffe model.

***doLoadTF***

`doLoadTF` method is to load a tensorflow model. The model can be loaded as a `FloatModel` or an `OpenVINOModel`. There are two backends to load a tensorflow model: TFNet and OpenVINO. 

<span id="jump">For OpenVINO backend, [supported tensorflow models](https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) are listed below:</span>

***doLoadOpenVINO***
                                          
`doLoadOpenVINO` method is to load an OpenVINO Intermediate Representation(IR).

***doLoadOpenVINOInt8***

`doLoadOpenVINOInt8` method is to load an OpenVINO Int8 Intermediate Representation(IR).

***doReload***

`doReload` method is to reload the bigdl, analytics-zoo model.

***doPredict***

`doPredict` method is to do the prediction.

***doPredictInt8***

`doPredict` method is to do the prediction with Int8 model. If model doesn't support predictInt8, will throw RuntimeException with `does not support predictInt8` message.

## Primary APIs for Python

**load**
`load` method is to load a bigdl, analytics-zoo model.

**load_caffe**
`load_caffe` method is to load a caffe model.

**load_openvino**
`load_openvino` method is to load an OpenVINO Intermediate Representation(IR).

**load_tf**
`load_tf` method is to load a tensorflow model. The model can be loaded as a `FloatModel` or an `OpenVINOModel`. There are two backends to load a tensorflow model: TFNet and OpenVINO.

**predict**
`predict` method is to do the prediction.

## Supportive classes

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


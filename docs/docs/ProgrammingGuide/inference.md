Inference Model is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It allows user to conveniently use pre-trained models from Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). Inference Model provides Java, Scala and Python interfaces.

**Highlights**

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
2. Support transformation of various input data type, thus supporting future prediction tasks.
3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).

## **Load and predict with pre-trained model**
**Basic usage of Inference Model:**

1. Directly use InferenceModel or write a subclass extends `InferenceModel` (`AbstractinferenceModel` in Java).
2. Load pre-trained models with corresponding `load` methods, e.g, `doLoad` for Zoo, and `doLoadTF` for TensorFlow.
3. Do prediction with `predict` method.

**Supported models:**

1. [Zoo Models](https://analytics-zoo.github.io/master/##built-in-deep-learning-models)
2. [Caffe Models](https://github.com/BVLC/caffe/wiki/Model-Zoo)
3. [TensorFlow Models](https://github.com/tensorflow/models)
4. [OpenVINO models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)

**Java**

Write a subclass that extends `AbstractinferenceModel`, implement or override methods. Then, load model with corresponding `load` methods (load Zoo, caffe, OpenVINO and TensorFlow model with `load`, `loadCaffe`, `doLoadCaffe` and `loadTF`), and do prediction with `predict` method. 

```java
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class ExtendedInferenceModel extends AbstractInferenceModel {
    public ExtendedInferenceModel() {
        super();
    }
}
ExtendedInferenceModel model = new ExtendedInferenceModel();
// Load Zoo model
model.load(modelPath, weightPath);
// Predict
List<List<JTensor>> result = model.predict(inputList);
```

**Scala**

New instance with `InferenceModel`, and load model with corresponding `load` methods (load Zoo, caffe, OpenVINO and TensorFlow model with `doLoad`, `doLoadCaffe`, `doLoadOpenVINO` and `doLoadTF`), then do prediction with `predict` method.

```scala
import com.intel.analytics.zoo.pipeline.inference.InferenceModel

val model = new InferenceModel()
// Load Zoo model
model.doLoad(modelPath, weightPath)
// Predict
val result = model.doPredict(inputList)
```

In some cases, you may want to write a subclass that extends `InferenceModel`, implement or override methods. Then, load model with corresponding `load` methods, and do prediction with `predict` method.

```scala
import com.intel.analytics.zoo.pipeline.inference.InferenceModel

class ExtendedInferenceModel extends InferenceModel {

}

val model = new ExtendedInferenceModel()
// Load Zoo model
model.doLoad(modelPath, weightPath)
// Predict
val result = model.doPredict(inputList)
```

**Python**

New instance `InferenceModel`, and load Zoo model with corresponding `load` methods (load Zoo, caffe, OpenVINO and TensorFlow model with `load`, `load_caffe`, `load_openvino` and `load_tf`), then do prediction with `predict` method.

```python
from zoo.pipeline.inference import InferenceModel

model = InferenceModel()
# Load Zoo model
model.load(modelPath, weightPath)
# Predict
result = model.predict(inputList)
```

In some cases, you may want to write a subclass that extends `InferenceModel`, implement or override methods. Then, load Zoo model with corresponding `doLoad` methods, and do prediction with `predict` method.

```python
from zoo.pipeline.inference import InferenceModel

class ExtendedInferenceModel(InferenceModel):

model = ExtendedInferenceModel()
# Load Zoo model
model.load(modelPath, weightPath)
# Predict
result = model.predict(inputList)
```

## **Examples**
We provide examples based on InferenceModel.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/model-inference-examples) for the Java example.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/streaming/textclassification) for the Scala example.

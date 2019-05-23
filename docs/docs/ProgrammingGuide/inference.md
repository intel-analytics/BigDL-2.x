Inference Model is a package in Analytics Zoo aiming to provide high level APIs to speed-up development. It allows user to conveniently use pre-trained models from Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR). Inference Model provides JAVA, Scala and Python interfaces.

**Highlights**

1. Easy-to-use APIs for loading and prediction with deep learning models of Analytics Zoo, Caffe, Tensorflow and OpenVINO Intermediate Representation(IR).
2. Support transformation of various input data type, thus supporting future prediction tasks.
3. Transparently support the OpenVINO toolkit, which deliver a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).

## **Load and predict with pre-trained model**
Basic usage of Inference Model:

1. Directly use InferenceModel or write a subclass extends `InferenceModel` (`AbstractinferenceModel` in JAVA).
2. Load pre-trained models with corresponding `load` methods, e.g, doLoad for Zoo, and doLoadTF for TensorFlow.
3. Do prediction with `predict` method.

**Java**
Write a subclass that extends `AbstractinferenceModel`, and load Zoo model with `load`, then do prediction with `predict`.
```java
import com.intel.analytics.zoo.pipeline.inference.AbstractInferenceModel;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class TextClassificationModel extends AbstractInferenceModel {
    public TextClassificationModel() {
        super();
    }
 }
TextClassificationModel model = new TextClassificationModel();
// Load Zoo, caffe, OpenVINO and TensorFlow model with load, loadCaffe, doLoadCaffe and loadTF
model.load(modelPath, weightPath);
// Predict
List<List<JTensor>> result = model.predict(inputList);
```

**Scala**
Write a subclass that extends `InferenceModel`, and load Zoo model with `doLoad`, then do prediction with `predict`.
```scala
import com.intel.analytics.zoo.pipeline.inference.InferenceModel

class TextClassificationModel extends InferenceModel {

}

val model = new TextClassificationModel()
// Load Zoo, caffe, OpenVINO and TensorFlow model with doLoad, doLoadCaffe, doLoadOpenVINO and doLoadTF
model.doLoad(modelPath, weightPath)
// Predict
val result = model.doPredict(inputList)
```

**Python**

Directly use `InferenceModel`, and load Zoo model with `load`, then do prediction with `predict`.
```python
from zoo.pipeline.inference import InferenceModel

model = InferenceModel()
# Load Zoo, caffe, OpenVINO and TensorFlow model with load, load_caffe, load_openvino and load_tf
model.load(modelPath, weightPath)
# Predict
result = model.predict(inputList)
```

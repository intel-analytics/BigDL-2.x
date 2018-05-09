## NNEstimator

**Scala:**

```scala
val estimator = class NNEstimator(model, criterion, samplePreprocessing)

**Python:**

```python
estimator = NNEstimator(model, criterion, samplePreprocessing)
```

[[NNEstimator]] extends [[org.apache.spark.ml.Estimator]] and supports training a BigDL
model with Spark DataFrame data. It can be integrated into a standard Spark ML Pipeline
to allow users combine the components of BigDL and Spark MLlib.

[[NNEstimator]] supports different feature and label data type through [[Preprocessing]]. We
provide pre-defined [[Preprocessing]] for popular data types like Array or Vector in package
[[com.intel.analytics.zoo.feature]], while user can also develop customized [[Preprocessing]].
During fit, NNEstimator will extract feature and label data from input DataFrame and use
the [[Preprocessing]] to prepare data for the model. Using the [[Preprocessing]] allows
[[NNEstimator]] to cache only the raw data and decrease the memory consumption during feature
conversion and training.
More concrete examples are available in package [[com.intel.analytics.zoo.examples.nnframes]]

Multiple constructors for [NNEstimator] are provided for different sceanarios.

**Scala Example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.zoo.feature.common._
import com.intel.analytics.zoo.pipeline.nnframes.NNEstimator

val model = Sequential().add(Linear(2, 2))
val criterion = MSECriterion()
val estimator = new NNEstimator(model, criterion, ArrayToTensor(Array(2)), ArrayToTensor(Array(2)))

//alternatively: new NNEstimator(model, criterion, FeatureLabelPreprocessing(ArrayToTensor(Array(2)), ArrayToTensor(Array(2))))
//alternatively: NNEstimator(model, criterion, Array(2), Array(2))
```

**Python Example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from zoo.pipeline.nnframes.nn_classifier import *
from zoo.feature.common import *

model = Sequential().add(Linear(2, 2))
criterion = MSECriterion()
estimator = NNEstimator(model, criterion, [2], [2]).setBatchSize(4).setMaxEpoch(10)

```
---


## NNModel
**Scala:**
```scala
val nnModel = new NNModel[T](model: Module[T], featureSize: Array[Int])
```

**Python:**
```python
nn_model = NNModel(model, feature_size)
```

`NNModel` is designed to wrap the BigDL Module as a Spark's ML [Transformer](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#transformers) which is compatible
with both spark 1.5-plus and 2.0. It greatly improves the
experience of Spark users because now you can **wrap a pre-trained BigDL Model into a NNModel,
and use it as a transformer in your Spark ML pipeline to predict the results**.

`NNModel` supports feature data in the format of
`Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
org.apache.spark.ml.linalg.{Vector, VectorUDT}` and image schema. Internally `DLModel` use
features column as storage of the feature data, and create Tensors according to the constructor
parameter featureSize.

* `model` fitted BigDL module to use in prediction
* `featureSize` The size (Tensor dimensions) of the feature data.
(e.g. an image may be with featureSize = 28 * 28)

---

## NNClassifier
**Scala:**
```scala
val classifer = new NNClassifer(model: Module[T], criterion: Criterion[T], val featureSize: Array[Int])
```

**Python:**
```python
classifier = NNClassifer(model, criterion, feature_size)
```

`NNClassifier` is a specialized `NNEstimator` that simplifies the data format for
classification tasks where the label space is discrete. It only supports label column of DoubleType or FloatType,
and the fitted `NNClassifierModel` will have the prediction column of DoubleType.

* `model` BigDL module to be optimized in the fit() method
* `criterion` the criterion used to compute the loss and the gradient
* `featureSize` The size (Tensor dimensions) of the feature data.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.nnframes.NNClassifier
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, LogSoftMax, Sequential}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val data = sc.parallelize(Seq(
      (Array(0.0, 1.0), 1.0),
      (Array(1.0, 0.0), 2.0),
      (Array(0.0, 1.0), 1.0),
      (Array(1.0, 0.0), 2.0)))
val df = sqlContext.createDataFrame(data).toDF("features", "label")
val model = Sequential().add(Linear(2, 2)).add(LogSoftMax())
val criterion = ClassNLLCriterion()
val estimator = new NNClassifier(model, criterion, Array(2))
  .setBatchSize(4)
  .setMaxEpoch(10)

val dlModel = estimator.fit(df)
dlModel.transform(df).show(false)
```

**Python Example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from bigdl.dlframes.dl_classifier import *
from pyspark.sql.types import *

#Logistic Regression with BigDL layers and Analytics zoo NNClassifier
model = Sequential().add(Linear(2, 2)).add(LogSoftMax())
criterion = ClassNLLCriterion()
estimator = NNClassifier(model, criterion, [2]).setBatchSize(4).setMaxEpoch(10)
data = sc.parallelize([
    ((0.0, 1.0), [1.0]),
    ((1.0, 0.0), [2.0]),
    ((0.0, 1.0), [1.0]),
    ((1.0, 0.0), [2.0])])

schema = StructType([
    StructField("features", ArrayType(DoubleType(), False), False),
    StructField("label", ArrayType(DoubleType(), False), False)])
df = sqlContext.createDataFrame(data, schema)
dlModel = estimator.fit(df)
dlModel.transform(df).show(False)
```

## NNClassifierModel ##

**Scala:**
```scala
val dlClassifierModel = new NNClassifierModel[T](model: Module[T], featureSize: Array[Int])
```

**Python:**
```python
dl_classifier_model = NNClassifierModel(model, feature_size)
```

NNClassifierModel extends DLModel, which is a specialized DLModel for classification tasks.
The prediction column will have the datatype of Double.

* `model` fitted BigDL module to use in prediction
* `featureSize` The size (Tensor dimensions) of the feature data. (e.g. an image may be with
featureSize = 28 * 28)
---

## Hyperparameter setting

Prior to the commencement of the training process, you can modify the batch size, the epoch number of your
training, and learning rate to meet your goal or NNEstimator/NNClassifier will use the default value.

Continue the codes above, NNEstimator and NNClassifier can be setted in the same way.

**Scala:**

```scala
//for esitmator
estimator.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01)
//for classifier
classifier.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01)
```
**Python:**

```python
# for esitmator
estimator.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01)
# for classifier
classifier.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01)

```

## Prepare the data and start the training process

Users need to convert the data into Spark's
[DataFrame/DataSet](https://spark.apache.org/docs/latest/sql-programming-guide.html#datasets-and-dataframes)
to feed to the NNEstimator/NNCLassifer.
Then after these steps, we can start training now.

Suppose `df` is the training data, simple call `fit` method and let Analytics Zoo train the model for you. You will
get a NNClassifierModel if you use NNClassifier.

**Scala:**

```scala
//get a NNClassifierModel
val nnClassifierModel = classifier.fit(df)
```

**Python:**

```python
# get a NNClassifierModel
nnClassifierModel = classifier.fit(df)
```
## Make prediction on chosen data by using NNClassifierModel

Since NNClassifierModel inherits from Spark's Transformer abstract class, simply call `transform`
 method on NNClassifierModel to make prediction.

**Scala:**

```scala
nnModel.transform(df).show(false)
```

**Python:**

```python
nnModel.transform(df).show(false)
```

For the complete examples of NNFrames, please refer to:
[Scala examples](https://github.com/intel-analytics/zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/nnframes/)
[Python examples]()
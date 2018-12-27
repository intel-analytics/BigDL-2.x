This page shows how to train, evaluate or predict a model using the Keras-Style API.

You may refer to the `User Guide` page to see how to define a model in [Python](../keras-api-python/) or [Scala](../keras-api-scala/) correspondingly.

You may refer to [`Layers`](../Layers/core/) section to find all the available layers.

After defining a model with the Keras-Style API, you can call the following __methods__ on the model:


---
## **Compile**

Configure the learning process. Must be called before [fit](#fit) or [evaluate](#evaluate).

**Scala:**
```scala
compile(optimizer, loss, metrics=null)
```
**Python**
```python
compile(optimizer, loss, metrics=None)
```

Parameters:

* `optimizer`: Optimization method to be used. One can alternatively pass in the corresponding string representation, such as 'sgd'.
* `loss`: Criterion to be used. One can alternatively pass in the corresponding string representation, such as 'mse'. (see [here](objectives/#available-objectives)).
* `metrics`: List of validation methods to be used. Default is None if no validation is needed. One can alternatively `Array("accuracy")`(Scala) `["accuracy"]`(Python).

---
## **Fit**

Train a model for a fixed number of epochs on a DataSet.

**Scala:**
```scala
fit(x, y=null, batch_size=32，nbEpoch=10, validationData=null, distributed=true)
```
**Python**
```python
fit(x, y=None, batch_size=32, nb_epoch=10, validation_data=None, distributed=True)
```

Parameters:

* `x`: Input data. A Numpy array or RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `y`: Labels. A Numpy array. Default is None if x is already Sample RDD or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `batch_Size`: Number of samples per gradient update. Default is 32.
* `nb_epoch`: Number of epochs to train.
* `validationData`: Tuple (x_val, y_val) where x_val and y_val are both Numpy arrays.
                    Can also be RDD of Sample or ImageSet or TextSet.
                    Default is None if no validation is involved.
* `distributed`: Boolean. Whether to train the model in distributed mode or local mode.
                 Default is True. In local mode, x and y must both be Numpy arrays.

---
## **Evaluate**

Evaluate a model on a given dataset in distributed mode.

**Scala:**
```scala
evaluate(x, y=null, batch_size=32)
```
**Python**
```python
evaluate(x, y=None, batch_size=32)
```

Parameters:

* `x`: Evaluation data. A Numpy array or RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `y`: Labels. A Numpy array. Default is None if x is already Sample RDD or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `batchSize`: Number of samples per batch. Default is 32.

---
## **Predict**

Use a model to do prediction.

**Scala:**
```scala
predict(x, distributed=True)
```
**Python**
```python
predict(x, distributed=True)
```

Parameters:

* `x`: Prediction data. A Numpy array or RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `batch_per_thread`:
        The default value is 4.
        When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.
        When distributed is False the total batch size is batch_per_thread * numOfCores.
* `distributed`: Boolean. Whether to do prediction in distributed mode or local mode.
                 Default is True. In local mode, x must be a Numpy array.
                 
## **Predict_classes**

Use a model to do prediction.

**Scala:**
```scala
predict_classes(x, batch_per_thread=4, zero_based_label=True)
```
**Python**
```python
predict_classes(x, batch_per_thread=4, zero_based_label=True)
```

Parameters:

* `x`: Prediction data. A Numpy array or RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text)..
* `batch_per_thread`:
        The default value is 4.
        When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.
        When distributed is False the total batch size is batch_per_thread * numOfCores.
* `zero_based_label`: Boolean. Whether result labels start from 0.
                      Default is True. If False, result labels start from 1.


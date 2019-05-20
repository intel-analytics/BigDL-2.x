# TFEstimator Class API

TFEstimator wraps a model defined by `model_fn`. The `model_fn` is almost identical to TensorFlow's `model_fn`
except users are required to return a `TFEstimator` object. Users do not need to construct backward graph
(calling `optimizer.minimize(...)`) but set a `loss` tensor in `TFEstimator`.

**Create a TFEstimator**:
```python
import tensorflow as tf
from zoo.tfpark.estimator import TFEstimator, TFEstimatorSpec
def model_fn(features, labels, mode):

    hidden = tf.layers.dense(features, 32, activation=tf.nn.relu)
    
    logits = tf.layers.dense(hidden, 10)

    if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
        return TFEstimatorSpec(mode, predictions=logits, loss=loss)
    else:
        return TFEstimatorSpec(mode, predictions=logits)

estimator = TFEstimator(model_fn, tf.train.AdamOptimizer(), model_dir="/tmp/estimator")
```

## Methods

### \_\_init\_\_

```python

```

#### Arguments


### train

```python
train(input_fn, steps=None)
```

#### Arguments

* **input_fn**: A function that constructs the input data for evaluation. The
            function should construct and return one of the following:
            
            * A `TFDataset` object, each elements of which is a tuple `(features, labels)`.
            * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
            `(features, labels)` with same constraints as below.
            * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
            of string feature name to `Tensor` and `labels` is a `Tensor` or a
            dictionary of string label name to `Tensor`. Both `features` and
            `labels` are consumed by `model_fn`. They should satisfy the expectation
            of `model_fn` from inputs.
* **steps**: Number of steps for which to train the model.


### evaluate

```python
evaluate(input_fn, eval_methods, steps=None, checkpoint_path=None)
```

#### Arguments

* **input_fn**: A function that constructs the input data for evaluation. The
            function should construct and return one of the following:
            
            * A `TFDataset` object, each elements of which is a tuple `(features, labels)`.
            * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
            `(features, labels)` with same constraints as below.
            * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
            of string feature name to `Tensor` and `labels` is a `Tensor` or a
            dictionary of string label name to `Tensor`. Both `features` and
            `labels` are consumed by `model_fn`. They should satisfy the expectation
            of `model_fn` from inputs.
* **eval_methods**: a list of strings to specify the evaluation metrics to
                    be used in this model
* **steps**: Number of steps for which to evaluate model.
* **checkpoint_path**: Path of a specific checkpoint to evaluate. If `None`, the
            latest checkpoint in `model_dir` is used.  If there are no checkpoints
            in `model_dir`, evaluation is run with newly initialized `Variables`
            instead of ones restored from checkpoint.

### predict

```python
predict(input_fn, checkpoint_path=None)
```

#### Arguments

* **input_fn**: A function that constructs the features.
              
              * A `TFDataset` object, each elements of which is a tuple `(features, None)`.
              * A `tf.data.Dataset` object: Outputs of `Dataset` object must have
                same constraints as below.
              * features: A `tf.Tensor` or a dictionary of string feature name to
                `Tensor`. features are consumed by `model_fn`. They should satisfy
                the expectation of `model_fn` from inputs.
              * A tuple, in which case the first item is extracted as features.

* **checkpoint_path**: Path of a specific checkpoint to predict. If `None`, the
            latest checkpoint in `model_dir` is used.  If there are no checkpoints
            in `model_dir`, prediction is run with newly initialized `Variables`
            instead of ones restored from checkpoint.



TFOptimizer is used for optimizing a TensorFlow model with respect to its training variables
on Spark/BigDL.

**Create a TFOptimizer**:
```python
import tensorflow as tf
from zoo.tfpark import TFOptimizer
optimizer = TFOptimizer.from_loss(loss, Adam(1e-3))
```

## Methods

### from_loss (factory method)

```python
from_loss(loss, optim_method, session=None, val_outputs=None,
                  val_labels=None, val_method=None, val_split=0.0,
                  clip_norm=None, clip_value=None, metrics=None,
                  tensor_with_value=None, **kwargs)
```

#### Arguments


* **loss**: The loss tensor of the TensorFlow model, should be a scalar
        
* **optim_method**: the optimization method to be used, such as bigdl.optim.optimizer.Adam
* **session**: the current TensorFlow Session, if you want to used a pre-trained model,
             you should use the Session to load the pre-trained variables and pass it to TFOptimizer.
* **val_outputs**: the validation output TensorFlow tensor to be used be val_methods
* **val_labels**: the validation label TensorFlow tensor to be used be val_methods
* **val_method**: the BigDL val_method(s) to be used.
* **val_split**: Float between 0 and 1. Fraction of the training data to be used as
               validation data. 
* **clip_norm**: float >= 0. Gradients will be clipped when their L2 norm exceeds
               this value.
* **clip_value**: float >= 0. Gradients will be clipped when their absolute value
                exceeds this value.
* **metrics**: a dictionary. The key should be a string representing the metric's name
             and the value should be the corresponding TensorFlow tensor, which should be a scalar.
* **tensor_with_value**: a dictionary. The key is TensorFlow tensor, usually a
                      placeholder, the value of the dictionary is a tuple of two elements. The first one of
                      the tuple is the value to feed to the tensor in training phase and the second one
                      is the value to feed to the tensor in validation phase.
* **kwargs**: used for backward compatibility


### from_keras (factory method)

```python
from_keras(keras_model, dataset, optim_method=None, val_spilt=0.0, **kwargs)
```

#### Arguments

* **keras_model**: the tensorflow.keras model, which must be compiled.
* **dataset**: a TFDataset
* **optim_method**: the optimization method to be used, such as bigdl.optim.optimizer.Adam
* **val_spilt**: Float between 0 and 1. Fraction of the training data to be used as
      validation data. 
* **kwargs**: used for backward compatibility


### set_train_summary

```python
set_train_summary(summary)
```

#### Arguments

* **summary**: The train summary to be set. A TrainSummary object contains information
               necessary for the optimizer to know how often the logs are recorded,
               where to store the logs and how to retrieve them, etc. For details,
               refer to the docs of bigdl.optim.optimizer.TrainSummary.

### set_val_summary

```python
set_val_summary(summary)
```

#### Arguments

* **summary**: The validation summary to be set. A ValidationSummary object contains information
               necessary for the optimizer to know how often the logs are recorded,
               where to store the logs and how to retrieve them, etc. For details,
               refer to the docs of bigdl.optim.optimizer.ValidationSummary.
               

### set_constant_gradient_clipping

```python
set_constant_gradient_clipping(min_value, max_value)
```

#### Arguments

* **min_value**: the minimum value to clip by
* **max_value**: the maxmimum value to clip by


### set_gradient_clipping_by_l2_norm

```python
set_gradient_clipping_by_l2_norm(self, clip_norm)
```

#### Arguments

* **clip_norm**: gradient L2-Norm threshold


### optimize

```python
optimize(self, end_trigger=None)
```

#### Arguments

* **end_trigger**: BigDL's Trigger to indicate when to stop the training. If none, defaults to
                   train for one epoch.




## Usage of objectives

An objective function (or loss function, or optimization score function) is one of the two parameters required to compile a model:

**Scala:**

```scala"
model.compile(loss = "mean_squared_error", optimizer = "sgd")
```

**Python:**

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.MeanSquaredError
model.compile(loss = MeanSquaredError(), optimizer = "sgd")
```

**Python:**

```python
from zoo.pipeline.api.keras import objectives.MeanSquaredError
model.compile(loss=MeanSquaredError(), optimizer='sgd')
```

---

## Available objectives

### MeanSquaredError

The mean squared error criterion e.g. input: a, target: b, total elements: n

```
loss(a, b) = 1/n * sum(|a_i - b_i|^2)
```

Parameters:

 * `sizeAverage` a boolean indicating whether to divide the sum of squared error by n. 
                 Default: true

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.MeanSquaredError
loss = MeanSquaredError()
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.MeanSquaredError
loss = MeanSquaredError()
```

### MeanAbsoluteError

Measures the mean absolute value of the element-wise difference between input and target

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.MeanAbsoluteError
loss = MeanAbsoluteError()
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.MeanAbsoluteError
loss = MeanAbsoluteError()
```

### MeanAbsolutePercentageError

Compute mean absolute percentage error for intput and target

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.MeanAbsolutePercentageError
loss = MeanAbsolutePercentageError()
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.MeanAbsolutePercentageError
loss = MeanAbsolutePercentageError()
```

### MeanSquaredLogarithmicError

Compute mean squared logarithmic error for input and target

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.MeanSquaredLogarithmicError
loss = MeanSquaredLogarithmicError()
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.MeanSquaredLogarithmicError
loss = MeanSquaredLogarithmicError()
```

### BinaryCrossEntropy

Also known as logloss. 

Parameters:

* `weights` A tensor assigning weight to each of the classes
* `sizeAverage` whether to divide the sequence length. Default is true.

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.BinaryCrossEntropy
loss = keras.objectives.BinaryCrossEntropy(weights, sizeAverage)
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.BinaryCrossEntropy
loss = BinaryCrossEntropy(weights, sizeAverage)
```

### CategoricalCrossEntropy

This is same with cross entropy criterion, except the target tensor is a
one-hot tensor.

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.CategoricalCrossEntropy
loss = CategoricalCrossEntropy()
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.CategoricalCrossEntropy
loss = CategoricalCrossEntropy()
```

### Hinge

Creates a criterion that optimizes a two-class classification hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.

Parameters:

 * `margin` if unspecified, is by default 1.
 * `sizeAverage` whether to average the loss, is by default true

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.Hinge
loss = keras.objectives.Hinge(margin=1.0, sizeAverage=true)
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.Hinge
loss = Hinge(margin=1.0, sizeAverage=True)
```

### RankHinge

Hinge loss for pairwise ranking problems.

Parameters:

 * `margin` if unspecified, is by default 1.

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.RankHinge
loss = RankHinge(margin=1.0)
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.RankHinge
loss = RankHinge(margin=1.0)
```

### SquaredHinge

Creates a criterion that optimizes a two-class classification squared hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.

Parameters:

 * `margin` if unspecified, is by default 1.
 * `sizeAverage` whether to average the loss, is by default true

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SquaredHinge
loss = SquaredHinge(margin=1.0, sizeAverage=true)
```

**Python:**

```python
from zoo.pipeline.api import keras
loss = SquaredHinge(margin=1.0, sizeAverage=true)
```

### Poisson

Compute Poisson error for intput and target

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.Poisson
loss = Poisson()
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.Poisson
loss = Poisson()
```

### CosineProximity

Computes the negative of the mean cosine proximity between predictions and targets.

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.CosineProximity
loss = CosineProximity()
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.CosineProximity
loss = CosineProximity()
```

### SparseCategoricalCrossEntropy

A loss often used in multi-class classification problems with SoftMax as the last layer of the neural network. By default, input(y_pred) is supposed to be probabilities of each class, and target(y_true) is supposed to be the class label starting from 0.

Parameters:

 * `logProbAsInput` Boolean. Whether to accept log-probabilities or probabilities as input. Default is false and inputs should be probabilities.
 * `zeroBasedLabel` Boolean. Whether target labels start from 0. Default is true. If false, labels start from 1.
 * `weights` Tensor. Weights of each class if you have an unbalanced training set. Default is null.
 * `sizeAverage` Boolean. Whether losses are averaged over observations for each mini-batch. Default is true. If false, the losses are instead summed for each mini-batch.
 * `paddingValue` Integer. If the target is set to this value, the training process will skip this sample. In other words, the forward process will return zero output and the backward process will also return zero gradInput. Default is -1.

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
loss = SparseCategoricalCrossEntropy()
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.SparseCategoricalCrossEntropy
loss = SparseCategoricalCrossEntropy()
```

### KullbackLeiblerDivergence

Loss calculated as:
```
y_true = K.clip(y_true, K.epsilon(), 1)
y_pred = K.clip(y_pred, K.epsilon(), 1)
```
and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras.objectives.KullbackLeiblerDivergence
loss = KullbackLeiblerDivergence()
```

**Python:**

```python
from zoo.pipeline.api import keras.objectives.KullbackLeiblerDivergence
loss = KullbackLeiblerDivergence()
```

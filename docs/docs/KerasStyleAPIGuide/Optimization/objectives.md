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
import com.intel.analytics.zoo.pipeline.api.keras.objectives
model.compile(loss = objectives.MeanSquaredError(), optimizer = "sgd")
```

**Python:**

```python
from zoo.pipeline.api.keras import objectives
model.compile(loss=objectives.MeanSquaredError(), optimizer='sgd')
```

---

## Available objectives

### MeanSquaredError

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.MeanSquaredError()
loss = keras.objectives.MSE()
loss = keras.objectives.mse()
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.MeanSquaredError()
keras.objectives.MSE()
keras.objectives.mse()
```

The mean squared error criterion e.g. input: a, target: b, total elements: n

```
loss(a, b) = 1/n * sum(|a_i - b_i|^2)
```

Parameters:

 * `sizeAverage` a boolean indicating whether to divide the sum of squared error by n. 
                 Default: true

### MeanAbsoluteError

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.MeanAbsoluteError()
loss = keras.objectives.MAE()
loss = keras.objectives.mae()
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.MeanAbsoluteError()
keras.objectives.MAE()
keras.objectives.mae()
```

Measures the mean absolute value of the element-wise difference between input and target

### MeanAbsolutePercentageError

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.MeanAbsolutePercentageError()
loss = keras.objectives.MAPE()
loss = keras.objectives.mape()
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.MeanAbsolutePercentageError()
keras.objectives.MAPE()
keras.objectives.mape()
```

compute mean absolute percentage error for intput and target

### MeanSquaredLogarithmicError

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.MeanSquaredLogarithmicError()
loss = keras.objectives.MSLE()
loss = keras.objectives.msle()
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.MeanSquaredLogarithmicError()
keras.objectives.MSLE()
keras.objectives.msle()
```

compute mean squared logarithmic error for input and target


### SquaredHinge

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.SquaredHinge(margin=1.0, sizeAverage=true)
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.SquaredHinge(margin=1.0, sizeAverage=true)
```

Creates a criterion that optimizes a two-class classification squared hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.

Parameters:

 * `margin` if unspecified, is by default 1.
 * `sizeAverage` whether to average the loss, is by default true

### Hinge

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.Hinge(margin=1.0, sizeAverage=true)
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.Hinge(margin=1.0, sizeAverage=True)
```

Creates a criterion that optimizes a two-class classification hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.

Parameters:

 * `margin` if unspecified, is by default 1.
 * `sizeAverage` whether to average the loss, is by default true

### BinaryCrossEntropy

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.BinaryCrossEntropy(weights, sizeAverage)
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.BinaryCrossEntropy(weights, sizeAverage)
```

Also known as logloss. 

Parameters:

* `weights` A tensor assigning weight to each of the classes
* `sizeAverage` whether to divide the sequence length. Default is true.

### CategoricalCrossEntropy

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.CategoricalCrossEntropy()
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.CategoricalCrossEntropy()
```

This is same with cross entropy criterion, except the target tensor is a
one-hot tensor.

### Poisson

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.Poisson()
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.Poisson()
```

compute Poisson error for intput and target

### CosineProximity

**Scala:**

```scala
import com.intel.analytics.zoo.pipeline.api.keras
loss = keras.objectives.CosineProximity()
```

**Python:**

```python
from zoo.pipeline.api import keras
keras.objectives.CosineProximity()
```

Computes the negative of the mean cosine proximity between predictions and targets.

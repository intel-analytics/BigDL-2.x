## Usage of objectives

An objective function (or loss function, or optimization score function) is one of the two parameters required to compile a model:

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

```python
from zoo.pipeline.api.keras import objectives
model.compile(loss=objectives.MeanSquaredError, optimizer='sgd')
```

---

## Available objectives

- __MeanSquaredError__ / __MSE__ / __mse__

```python
from zoo.pipeline.api import keras
keras.objectives.MeanSquaredError()
keras.objectives.MSE()
keras.objectives.mse()
```

- __MeanAbsoluteError__ / __MAE__ / __mae__

```python
from zoo.pipeline.api import keras
keras.objectives.MeanAbsoluteError()
keras.objectives.MAE()
keras.objectives.mae()
```

- __MeanAbsolutePercentageError__ / __MAPE__ /__mape__

```python
from zoo.pipeline.api import keras
keras.objectives.MeanAbsoluteError()
keras.objectives.MAPE()
keras.objectives.mape()
```

- __MeanSquaredLogarithmicError__ / __MSLE__ / __msle__

```python
from zoo.pipeline.api import keras
keras.objectives.MeanSquaredLogarithmicError()
keras.objectives.MSLE()
keras.objectives.msle()
```

- __SquaredHinge__

```python
from zoo.pipeline.api import keras
keras.objectives.SquaredHinge()
```

- __Hinge__

```python
from zoo.pipeline.api import keras
keras.objectives.Hinge()
```

- __BinaryCrossEntropy__: Also known as logloss. 

```python
from zoo.pipeline.api import keras
keras.objectives.BinaryCrossEntropy()
```

- __CategoricalCrossEntropy__: Also known as multiclass logloss. __Note__: using this objective requires that your labels are binary arrays of shape `(nb_samples, nb_classes)`.

```python
from zoo.pipeline.api import keras
keras.objectives.CategoricalCrossEntropy()
```

- __Poisson__: mean of `(predictions - targets * log(predictions))`

```python
from zoo.pipeline.api import keras
keras.objectives.Poisson()
```

- __CosineProximity__: the opposite (negative) of the mean cosine proximity between predictions and targets.

```python
from zoo.pipeline.api import keras
keras.objectives.CosineProximity()
```

## **WithinChannelLRN2D**
The local response normalization layer performs a kind of "lateral inhibition" by normalizing over local input regions. The local regions extend spatially, in separate channels (i.e., they have shape 1 x size x size).

When you use this layer as the first layer of a model, you need to provide the argument inputShape (a Single Shape, does not include the batch dimension).

Remark: This layer is from Torch and wrapped in Keras style.

**Scala:**
```scala
WithinChannelLRN2D(size=5, alpha=1.0, beta=0.75, inputShape=(2, 3, 8, 8))
```
**Python:**
```python
WithinChannelLRN2D(size=5, alpha=1.0, beta=0.75, input_shape=(2, 3, 8, 8))
```

**Parameters:**

* 'size': The side length of the square region to sum over. Default is 5.
* 'alpha': The scaling parameter. Default is 1.0.
* 'beta': The exponent. Default is 0.75.
* 'input_shape': A shape tuple, not including batch.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.WithinChannelLRN2D
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(WithinChannelLRN2D[Float](inputShape = Shape(2, 3, 8)))
val input = Tensor[Float](1, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.11547339     -0.52518076     0.22743009      0.24847448
-0.72996384     1.5127875       1.285603        -0.8665928
2.2911248       0.062601104     -0.07974513     -0.26207858

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
-0.089576244    -0.39988548     0.17317083      0.21585277
-0.5662553      1.1518734       0.97888964      -0.7528196
1.7772957       0.047666013     -0.060719892    -0.22767082

[com.intel.analytics.bigdl.tensor.DenseTensor of size 1x3x4]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.layers import WithinChannelLRN2D
from zoo.pipeline.api.keras.models import Sequential

model = Sequential()
model.add(WithinChannelLRN2D(input_shape=(2, 3, 8)))
input = np.random.random([1, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.96982874, 0.80581477, 0.35435895, 0.45537825],
  [0.61421818, 0.54708709, 0.86205409, 0.07374387],
  [0.67227822, 0.25118575, 0.36258901, 0.28671433]]]
```
Output is
```python
[[[0.87259495, 0.71950066, 0.3164021 , 0.42620906],
  [0.55263746, 0.48848635, 0.76971596, 0.06902022],
  [0.60487646, 0.22428022, 0.32375062, 0.26834887]]]
```

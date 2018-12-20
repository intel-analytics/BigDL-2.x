## **PReLU**
Applies parametric ReLU, where parameter varies the slope of the negative part.

It follows: f(x) = max(0, x) + a * min(0, x)

**Scala:**
```scala
PReLU(nOutputPlane = 0, inputShape = null)
```
**Python:**
```python
PReLU(nOutputPlane=0, input_shape=None)
```

**Parameters:**

* `nOutputPlane`: Input map number. Default is 0,
                  which means using PReLU in shared version and has only one parameter.
* `inputShape`:  A Single Shape, does not include the batch dimension.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.PReLU
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(PReLU[Float](inputShape = Shape(3)))
val input = Tensor[Float](2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
-0.9026888      -1.0402212      1.3878769
-0.17167428     0.08202032      1.2682742
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
-0.2256722      -0.2600553      1.3878769
-0.04291857     0.08202032      1.2682742
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import PReLU

model = Sequential()
model.add(PReLU(input_shape=(3, )))
input = np.random.random([2, 3])
output = model.forward(input)
```
Input is:
```python
[[0.61639702 0.08877075 0.93652509]
 [0.38800821 0.76286851 0.95777973]]
```
Output is
```python
[[0.616397   0.08877075 0.9365251 ]
 [0.3880082  0.7628685  0.9577797 ]]
```
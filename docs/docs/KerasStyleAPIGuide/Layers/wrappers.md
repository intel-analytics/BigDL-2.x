## **InternalTimeDistributed**
InternalTimeDistributed wrapper. This layer is intended to apply contained layer to each temporal time slice of input tensor.

The input data format is [Batch, Time, Other dims]. For the contained layer, it must not change the Other dims length.

**Scala:**
```scala
InternalTimeDistributed(layer, maskZero = false)
```
**Python:**
```python
InternalTimeDistributed(layer, input_shape=None, name=None)
```

Parameters:

* `layer`: A layer instance.
* `maskZero`: For Scala API, if `maskZero` is set to true, if the input including zero vectors, the
corresponding output will be set to zero vecotrs.
For Python API, 

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalTimeDistributed
import com.intel.analytics.bigdl.nn.keras.Dense
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath

val model = Sequential[Float]()
model.add(InternalTimeDistributed(Dense(8, activation = "relu"), maskZero = false))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-0.15650798	-0.60011286	-0.0883946
-0.8020574	-2.0070791	0.58417106

(2,.,.) =
1.1210757	0.061217457	0.37585327
0.11572507	0.045938224	-1.1890792

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.35345355	0.019948795 0.0	        0.22901565	 0.0  0.035260748  0.0	        0.40403664
1.4793522	0.803728	0.0	        0.93547887	 0.0  0.097175285  0.0	        1.2386305

(2,.,.) =
0.06176605	0.0	        0.051847294 0.76588714   0.0  0.67298067   0.10942559   0.0
0.0	        0.0	        0.0	        0.0	         0.0  0.0	       0.4285032    0.3072814

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x8]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
//?from zoo.pipeline.api.keras.layers.internal import InternalTimeDistributed
from bigdl.nn.keras.layer import Dense

model = Sequential()
model.add(InternalTimeDistributed(Dense(8, activation = "relu")))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[0.37107995 0.16777911 0.07691505]
  [0.42678424 0.53602176 0.01580607]]

 [[0.31664302 0.03947526 0.1556008 ]
  [0.2834384  0.68845104 0.23020768]]]
```
Output is:
```python
[[[0.09678233 0.21351711 0.0   0.07420383 0.09885262 0.0 0.13514107 0.0 ]
  [0.06882857 0.18277436 0.0   0.1371126  0.00853634 0.0 0.1224944  0.0 ]]

 [[0.11387025 0.20642482 0.0   0.04896355 0.11478973 0.0 0.12610494 0.0 ]
  [0.08322716 0.08292685 0.0   0.14674747 0.0        0.0 0.05299555 0.0 ]]]
```

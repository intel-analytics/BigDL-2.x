## **SpatialDropout3D**
Spatial 3D version of Dropout.

This version performs the same function as Dropout, however it drops entire 3D feature maps instead of individual elements. If adjacent voxels within feature maps are strongly correlated (as is normally the case in early convolution layers) then regular dropout will not regularize the activations and will otherwise just result in an effective learning rate decrease. In this case, SpatialDropout3D will help promote independence between feature maps and should be used instead.

The input of this layer should be 5D.

**Scala:**
```scala
SpatialDropout3D(p = 0.5, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
SpatialDropout3D(p=0.5, dim_ordering="th", input_shape=None, name=None)
```

**Parameters:**

* `p`: Fraction of the input units to drop. Between 0 and 1.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.SpatialDropout3D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(SpatialDropout3D[Float](inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
1.5842006       -1.486708       -1.0261744
-0.8227147      0.1386223       -0.46191332

(1,1,2,.,.) =
-0.7794714      0.52259976      1.5326598
0.32597166      0.84018683      -0.24034925

(1,2,1,.,.) =
0.5037644       -0.42065156     1.1590574
1.4855213       -1.4098096      0.5154563

(1,2,2,.,.) =
2.1119535       0.4159602       -0.33109334
-1.9544226      0.014503485     -0.7715549

(2,1,1,.,.) =
1.1496683       0.20273614      -2.6363356
-1.6820912      -1.1656585      -0.8387814

(2,1,2,.,.) =
-1.1125584      -1.9073812      0.78532314
-1.0033096      -0.24038585     1.0534006

(2,2,1,.,.) =
0.46944886      -1.8767697      0.7275591
0.36211884      0.34403932      -1.3721423

(2,2,2,.,.) =
0.37117565      -0.45195773     0.66517854
0.3873176       -1.8218406      1.9105781

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
0.0     -0.0    -0.0
-0.0    0.0     -0.0

(1,1,2,.,.) =
-0.0    0.0     0.0
0.0     0.0     -0.0

(1,2,1,.,.) =
0.0     -0.0    0.0
0.0     -0.0    0.0

(1,2,2,.,.) =
0.0     0.0     -0.0
-0.0    0.0     -0.0

(2,1,1,.,.) =
0.0     0.0     -0.0
-0.0    -0.0    -0.0

(2,1,2,.,.) =
-0.0    -0.0    0.0
-0.0    -0.0    0.0

(2,2,1,.,.) =
0.0     -0.0    0.0
0.0     0.0     -0.0

(2,2,2,.,.) =
0.0     -0.0    0.0
0.0     -0.0    0.0

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import SpatialDropout3D

model = Sequential()
model.add(SpatialDropout3D(input_shape=(2, 2, 2, 2)))
input = np.random.random([2, 2, 2, 2, 2])
output = model.forward(input)
```
Input is:
```python
[[[[[0.19861794 0.32822715]
    [0.78735804 0.0586697 ]]

   [[0.22181565 0.09894792]
    [0.43668179 0.22321872]]]


  [[[0.81122679 0.44084158]
    [0.70199098 0.10383273]]

   [[0.78102397 0.62514588]
    [0.6933126  0.7830806 ]]]]



 [[[[0.22229716 0.90939922]
    [0.2453606  0.49500498]]

   [[0.95518136 0.78983711]
    [0.724247   0.62801332]]]


  [[[0.89800761 0.5523274 ]
    [0.83153558 0.58200981]]

   [[0.84787731 0.16651971]
    [0.22528241 0.68706778]]]]]
```
Output is
```python
[[[[[0.19861795 0.32822713]
    [0.78735805 0.0586697 ]]

   [[0.22181565 0.09894791]
    [0.43668178 0.22321871]]]


  [[[0.8112268  0.4408416 ]
    [0.70199096 0.10383273]]

   [[0.781024   0.62514585]
    [0.6933126  0.7830806 ]]]]



 [[[[0.         0.        ]
    [0.         0.        ]]

   [[0.         0.        ]
    [0.         0.        ]]]


  [[[0.89800763 0.5523274 ]
    [0.8315356  0.5820098 ]]

   [[0.8478773  0.16651972]
    [0.22528242 0.6870678 ]]]]]
```
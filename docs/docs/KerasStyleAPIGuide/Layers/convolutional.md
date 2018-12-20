## **LocallyConnected2D**
Locally-connected layer for 2D inputs that works similarly to the SpatialConvolution layer, except that weights are unshared, that is, a different set of filters is applied at each different patch of the input.

The input of this layer should be 4D.

**Scala:**
```scala
LocallyConnected2D(nbFilter, nbRow, nbCol, activation = null, borderMode = "valid", subsample = (1, 1), dimOrdering = "th", wRegularizer = null, bRegularizer = null, bias = true, inputShape = null)
```
**Python:**
```python
LocallyConnected2D(nb_filter, nb_row, nb_col, activation=None, border_mode="valid", subsample=(1, 1), dim_ordering="th", W_regularizer=None, b_regularizer=None, bias=True, input_shape=None, name=None)
```

**Parameters:**

* `nbFilter`: Number of convolution filters to use.
* `nbRow`: Number of rows in the convolution kernel.
* `nbCol`: Number of columns in the convolution kernel.
* `activation`: String representation of the activation function to use. See [here](activation/#available-activations) for available activation strings. Default is null.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `subsample`: Length 2 corresponding to the step of the convolution in the height and width dimension. Also called strides elsewhere. Default is (1, 1).
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `wRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), (eg. L1 or L2 regularization), applied to the input weights matrices. Default is null.
* `bRegularizer`: An instance of [Regularizer](../../APIGuide/Regularizers/), applied to the bias. Default is null.
* `bias`: Whether to include a bias (i.e. make the layer affine rather than linear). Default is true.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.LocallyConnected2D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(LocallyConnected2D[Float](2, 2, 2, inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
-0.71993834     0.018790463     0.08133635      0.35603827
-1.1757486      1.8503827       -1.4548069      -0.6309117
-0.53039306     -0.14174776     0.7653523       -0.1891388

(1,2,.,.) =
1.0949191       0.13689162      0.35839355      -0.14805469
-2.5264592      -0.34186792     1.3190275       -0.11725446
-0.48823252     -1.5305915      -1.0556486      1.792275

(2,1,.,.) =
0.92393816      0.83243525      0.22506136      0.6694662
0.7662836       -0.23876576     -0.7719174      0.13114463
0.042082224     1.2212821       -1.2496184      -0.18717249

(2,2,.,.) =
0.726698        0.42673108      0.0786712       -1.4069401
-0.090565465    0.49527475      0.08590904      -0.51858175
1.4575573       0.9669369       0.21832618      0.34654656

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,.,.) =
0.022375792     0.669761        -0.25723624
0.99919814      0.93189466      0.8592935

(1,2,.,.) =
0.12613812      -1.0531536      0.8148589
0.66276294      0.12609969      0.6590149

(2,1,.,.) =
-0.1259023      0.32203823      0.07248953
-0.125191       -0.1285046      0.021367729

(2,2,.,.) =
-0.13560611     -0.038621478    -0.08420516
-0.0021556932   -0.094522506    -0.08551059

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import LocallyConnected2D

model = Sequential()
model.add(LocallyConnected2D(2, 2, 2, input_shape=(2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.75179142 0.10678918 0.92663152 0.2041142 ]
   [0.03534582 0.13742629 0.94115987 0.17303432]
   [0.91112368 0.19837546 0.45643767 0.16589123]]

  [[0.22996923 0.22878544 0.75623624 0.7058976 ]
   [0.14107232 0.49484648 0.71194356 0.53604538]
   [0.46257205 0.46902871 0.48046811 0.83579709]]]


 [[[0.9397535  0.51814825 0.10492714 0.24623405]
   [0.69800376 0.12353963 0.69536497 0.05159074]
   [0.56722731 0.33348394 0.47648031 0.25398067]]

  [[0.51018599 0.3416568  0.14112375 0.76505795]
   [0.16242231 0.16735028 0.79000471 0.98701885]
   [0.79852431 0.77458166 0.12551857 0.43866238]]]]
```
Output is
```python
[[[[ 0.14901309 -0.11168094  0.28349853]
   [ 0.21792562  0.49922782 -0.06560349]]

  [[ 0.6176302  -0.4638375  -0.13387583]
   [-0.04903107  0.07764787 -0.33653474]]]


 [[[ 0.24676235 -0.46874076  0.33973938]
   [ 0.21408634  0.36619198  0.17972258]]

  [[ 0.35941058 -0.23446569 -0.09271184]
   [ 0.39490524 -0.00668371 -0.25355732]]]]
```
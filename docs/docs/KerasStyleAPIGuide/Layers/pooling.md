## **MaxPooling1D**
Max pooling operation for temporal data.

The input of this layer should be 3D.

**Scala:**
```scala
MaxPooling1D(poolLength = 2, stride = -1, borderMode = "valid", inputShape = null)
```
**Python:**
```python
MaxPooling1D(pool_length=2, stride=None, border_mode="valid", input_shape=None, name=None)
```

Parameters:

* `poolLength`: Size of the region to which max pooling is applied. Integer. Default is 2.
* `stride`: Factor by which to downscale. 2 will halve the input. If not specified, it will default to poolLength.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.layers.MaxPooling1D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

val model = Sequential[Float]()
model.add(MaxPooling1D[Float](poolLength = 3, inputShape = Shape(4, 5)))
val input = Tensor[Float](3, 4, 5).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
-1.2339195      -1.2134796      0.16991705      -0.10169973     -1.2464932
0.37946555      0.29533234      -1.2552645      -2.6928735      -0.44519955
0.98743796      -1.0912303      -0.13897413     1.0241779       -0.5951304
-0.31459442     -0.088579334    -0.58336115     -0.6427486      -0.1447043

(2,.,.) =
0.14750746      0.07493488      -0.8554524      -1.6551514      0.16679412
-0.82279974     0.25704315      0.09921734      -0.8135057      2.7640774
-1.0111052      0.34388593      -0.7569789      1.0547938       1.6738676
0.4396624       -1.0570261      0.061429325     1.1752373       -0.14648575

(3,.,.) =
-0.95818335     0.8790822       -0.99111855     -0.9717616      -0.39238095
1.2533073       0.23365906      1.7784269       1.0600376       1.6816885
0.7145845       0.4711851       -0.4465603      -0.77884597     0.484986
0.42429695      -2.00715        0.6520644       1.3022201       -0.48169184


[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x4x5]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,.,.) =
0.98743796      0.29533234      0.16991705      1.0241779       -0.44519955

(2,.,.) =
0.14750746      0.34388593      0.09921734      1.0547938       2.7640774

(3,.,.) =
1.2533073       0.8790822       1.7784269       1.0600376       1.6816885

[com.intel.analytics.bigdl.tensor.DenseTensor of size 3x1x5]

```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import MaxPooling1D

model = Sequential()
model.add(MaxPooling1D(pool_length = 3, input_shape = (4, 5)))
input = np.random.random([3, 4, 5])
output = model.forward(input)
```
Input is:
```python
[[[0.14508341 0.42297648 0.50516337 0.15659868 0.83121192]
  [0.27837702 0.87282932 0.94292864 0.48428998 0.23604637]
  [0.24147633 0.2116796  0.54433489 0.22961905 0.88685975]
  [0.57235359 0.16278372 0.39749189 0.20781401 0.22834635]]

 [[0.42306184 0.43404804 0.22141668 0.0316458  0.08445576]
  [0.88377164 0.00417697 0.52975728 0.43238725 0.40539813]
  [0.90702837 0.37940347 0.06435512 0.33566794 0.50049895]
  [0.12146178 0.61599986 0.11874934 0.57207512 0.87713768]]

 [[0.56690324 0.99869154 0.87789702 0.67840158 0.64935853]
  [0.9950283  0.55710408 0.70919634 0.52309929 0.14311439]
  [0.25394468 0.41519219 0.8074057  0.05341861 0.98447171]
  [0.71387206 0.74763239 0.27057394 0.09578605 0.68601852]]]
```
Output is:
```python
[[[0.27837703 0.8728293  0.9429287  0.48428997 0.8868598 ]]

 [[0.9070284  0.43404803 0.52975726 0.43238723 0.50049895]]

 [[0.9950283  0.99869156 0.877897   0.6784016  0.98447174]]]
```

---
## **MaxPooling2D**
Max pooling operation for spatial data.

The input of this layer should be 4D.

**Scala:**
```scala
MaxPooling2D(poolSize = (2, 2), strides = null, borderMode = "valid", dimOrdering = "th", inputShape = null)
```
**Python:**
```python
MaxPooling2D(pool_size=(2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `poolSize`: Length 2 corresponding to the downscale vertically and horizontally. Default is (2, 2), which will halve the image in each dimension.
* `strides`: Length 2. Stride values. Default is null, and in this case it will be equal to poolSize.
* `borderMode`: Either 'valid' or 'same'. Default is 'valid'.
* `dimOrdering`: Format of input data. Either 'th' (Channel First) or 'tf' (Channel Last). Default is 'th'.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.MaxPooling2D

val model = Sequential[Float]()
model.add(MaxPooling2D[Float](inputShape = Shape(2, 3, 4)))
val input = Tensor[Float](2, 2, 3, 4).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,.,.) =
0.02138003      -0.20666665     -0.93250555     0.41267508
-0.40883347     0.4919021       0.7189889       1.3442185
-0.08697278     -0.025719838    2.1126  0.69069535

(1,2,.,.) =
-0.1685801      -0.07843445     1.3499486       -0.5944459
0.29377022      0.061167963     -0.60608864     -0.08283464
0.03402891      -1.0627178      1.9463096       0.0011169242

(2,1,.,.) =
-1.4524128      1.3868454       2.3057284       1.574949
-1.165581       0.79445213      -0.63500565     -0.17981622
-0.98042095     -1.7876958      0.8024988       -0.90554804

(2,2,.,.) =
-1.6468426      1.1864686       -0.683854       -1.5643677
2.8272789       -0.5537863      -0.563258       -0.01623243
-0.31333938     0.03472893      -1.730748       -0.15463233

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3x4]
```
Output is:
```scala
output:
(1,1,.,.) =
0.4919021       1.3442185

(1,2,.,.) =
0.29377022      1.3499486

(2,1,.,.) =
1.3868454       2.3057284

(2,2,.,.) =
2.8272789       -0.01623243

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x1x2]
```

**Python example:**
```python
import numpy as np
from bigdl.nn.keras.topology import Sequential
from bigdl.nn.keras.layer import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(input_shape = (2, 3, 4)))
input = np.random.random([2, 2, 3, 4])
output = model.forward(input)
```
Input is:
```python
[[[[0.58589442 0.94643201 0.24779969 0.55347075]
   [0.50604116 0.69884915 0.81253572 0.58586743]
   [0.94560389 0.11573268 0.12562681 0.63301697]]

  [[0.11736968 0.75641404 0.19342809 0.37670404]
   [0.55561582 0.54354621 0.9506264  0.65929266]
   [0.72911388 0.00499644 0.24280364 0.28822998]]]


 [[[0.53249492 0.43969012 0.20407128 0.49541971]
   [0.00369797 0.75294821 0.15204289 0.41394393]
   [0.19416915 0.93034988 0.0358259  0.38001445]]

  [[0.88946341 0.30646232 0.5347175  0.87568066]
   [0.00439823 0.97792811 0.34842225 0.20433116]
   [0.42777728 0.93583737 0.54341935 0.31203758]]]]

```
Output is:
```python
[[[[0.946432   0.8125357 ]]

  [[0.75641406 0.95062643]]]


 [[[0.7529482  0.4954197 ]]

  [[0.9779281  0.8756807 ]]]]

```

---
## **AveragePooling3D**
Applies average pooling operation for 3D data (spatial or spatio-temporal).

Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').

Border mode currently supported for this layer is 'valid'.

The input of this layer should be 5D.

**Scala:**
```scala
AveragePooling3D(poolSize = (2, 2, 2), strides = null, dimOrdering = "th", inputShape = null)
```
**Python:**
```python
AveragePooling3D(pool_size=(2, 2, 2), strides=None, border_mode="valid", dim_ordering="th", input_shape=None, name=None)
```

Parameters:

* `poolSize`: Length 3. Factors by which to downscale (dim1, dim2, dim3). Default is (2, 2, 2), which will halve the image in each dimension.
* `strides`: Length 3. Stride values. Default is null, and in this case it will be equal to poolSize.
* `dimOrdering`: Format of input data. Only 'th' (Channel First) is supported for now.
* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.AveragePooling3D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(AveragePooling3D[Float](inputShape = Shape(2, 2, 2, 3)))
val input = Tensor[Float](2, 2, 2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,1,1,.,.) =
-0.71569425     -0.39595184     -0.47607258
-0.12621938     -0.66759187     0.86833215

(1,1,2,.,.) =
1.219894        -0.07514859     0.6606987
0.073906526     -1.2547257      -0.49249622

(1,2,1,.,.) =
-1.0730773      0.2780401       -0.8603222
-0.31499937     0.94786566      -1.6953986

(1,2,2,.,.) =
0.31038517      1.7660809       -0.9849316
-1.5245554      0.24002236      0.473947

(2,1,1,.,.) =
-0.988634       -0.0028023662   -2.1534977
0.58303267      0.72106487      0.22115333

(2,1,2,.,.) =
1.3964092       -0.59152335     -0.6552192
2.0191588       -0.32599944     0.84014076

(2,2,1,.,.) =
1.4505147       -2.4253457      -0.37597662
-0.7049585      1.3384854       -1.1081233

(2,2,2,.,.) =
-0.8498942      1.169977        0.78120154
0.13814813      -0.7438999      -0.9272572

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
(1,1,1,.,.) =
-0.24269137

(1,2,1,.,.) =
0.07872025

(2,1,1,.,.) =
0.3513383

(2,2,1,.,.) =
-0.078371644


[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x1x1x1]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import AveragePooling3D

model = Sequential()
model.add(AveragePooling3D(input_shape = (2, 2, 2, 3)))
input = np.random.random([2, 2, 2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[[[0.95796698 0.76067104 0.47285625]
    [0.90296063 0.64177821 0.23302549]]

   [[0.37135542 0.38455108 0.66999497]
    [0.06756778 0.16411331 0.39038159]]]


  [[[0.9884323  0.97861344 0.69852249]
    [0.53289779 0.51290587 0.54822396]]

   [[0.77241923 0.06470524 0.00757586]
    [0.65977832 0.31973607 0.7551191 ]]]]



 [[[[0.56819589 0.20398916 0.26409867]
    [0.81165023 0.65269175 0.16519667]]

   [[0.7350688  0.52442381 0.29116889]
    [0.45458689 0.29734681 0.39667421]]]


  [[[0.33577239 0.54035235 0.41285576]
    [0.01023886 0.23677996 0.18901205]]

   [[0.67638612 0.54170351 0.0068781 ]
    [0.95769069 0.88558419 0.4262852 ]]]]]
```
Output is:
```python
[[[[[0.5313706 ]]]


  [[[0.603686  ]]]]



 [[[[0.5309942 ]]]


  [[[0.52306354]]]]]

```

---
## **GlobalMaxPooling1D**
Global max pooling operation for temporal data.

The input of this layer should be 3D.

**Scala:**
```scala
GlobalMaxPooling1D(inputShape = null)
```
**Python:**
```python
GlobalMaxPooling1D(input_shape=None, name=None)
```

Parameters:

* `inputShape`: Only need to specify this argument when you use this layer as the first layer of a model. For Scala API, it should be a [`Shape`](../keras-api-scala/#shape) object. For Python API, it should be a shape tuple. Batch dimension should be excluded.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.GlobalMaxPooling1D
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor

val model = Sequential[Float]()
model.add(GlobalMaxPooling1D[Float](inputShape = Shape(2, 3)))
val input = Tensor[Float](2, 2, 3).randn()
val output = model.forward(input)
```
Input is:
```scala
input: com.intel.analytics.bigdl.tensor.Tensor[Float] =
(1,.,.) =
1.2998451       2.1855159       -0.05535197
-0.6448657      0.74119943      -0.8761581

(2,.,.) =
1.3994918       -1.5119147      -0.6625015
1.803635        -2.2516544      -0.016894706

[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2x3]
```
Output is:
```scala
output: com.intel.analytics.bigdl.nn.abstractnn.Activity =
1.2998451       2.1855159       -0.05535197
1.803635        -1.5119147      -0.016894706
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x3]
```

**Python example:**
```python
import numpy as np
from zoo.pipeline.api.keras.models import Sequential
from zoo.pipeline.api.keras.layers import GlobalMaxPooling1D

model = Sequential()
model.add(GlobalMaxPooling1D(input_shape = (2, 3)))
input = np.random.random([2, 2, 3])
output = model.forward(input)
```
Input is:
```python
[[[0.05589183 0.73674405 0.49270549]
  [0.03348098 0.82000941 0.81752936]]

 [[0.97310222 0.8878789  0.72330625]
  [0.86144601 0.88568162 0.47241316]]]
```
Output is:
```python
[[0.05589183 0.8200094  0.8175294 ]
 [0.9731022  0.8878789  0.72330624]]
```

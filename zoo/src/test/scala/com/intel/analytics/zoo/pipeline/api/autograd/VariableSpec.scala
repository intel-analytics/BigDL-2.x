/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.api.autograd

import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad => A}
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasBaseSpec
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

class VariableSpec extends KerasBaseSpec {

  "get Variable name" should "be test" in {
    val yTrue = Variable[Float](inputShape = Shape(3))
    val t = A.log(yTrue)
    val name = t.name
    assert(name.contains("Log"))
  }

  "Expand and CMulTable" should "be able to generate correct result" in {
    val xValue = Tensor[Float]((2 until 18).toArray.map(_.toFloat), Array(2, 2, 4))
    val yValue = Tensor[Float]((2 until 6).toArray.map(_.toFloat), Array(1, 4))
    val x = Variable[Float](inputShape = Shape(2, 4))
    val y = Parameter[Float](inputShape = Shape(1, 4),
      initWeight = yValue)
    val z = x * y
    val model = Model[Float](Array(x, y), z)
    val output = model.forward(T(xValue, yValue)).toTensor[Float]
    val expect = Tensor[Float](Array[Float](4f, 9f, 16f, 25f, 12f, 21f, 32f, 45f,
    20f, 33f, 48f, 65f, 28f, 45f, 64f, 85f), Array(2, 2, 4))
    require(output.almostEqual(expect, 1e-8) == true)

    val gradOutput = Tensor[Float]((20 until 36).toArray.map(_.toFloat), Array(2, 2, 4))
    val gradInput = model.backward(T(xValue, yValue), gradOutput).toTable
    val expect2 = Tensor[Float](Array[Float](40f, 63f, 88f, 115f, 48f, 75f, 104f, 135f,
    56f, 87f, 120f, 155f, 64f, 99f, 136f, 175f), Array(2, 2, 4))
    require(gradInput[Tensor[Float]](1).almostEqual(expect2, 1e-8) == true)
    val expect3 = Tensor[Float](Array[Float](912f, 1052f, 1200f, 1356f), Array(1, 4))
    require(gradInput[Tensor[Float]](2).almostEqual(expect3, 1e-8) == true)
  }

  "Variable operator" should "be able to work with different element size" in {
    val x = Variable[Float](inputShape = Shape(2, 5))
    val y = Variable[Float](inputShape = Shape(2, 1))
    val diff = x - y
    val model = Model[Float](Array(x, y), diff)
    val xValue = Tensor[Float](3, 2, 5).rand()
    val yValue = Tensor[Float](3, 2, 1).rand()
    val output = model.forward(T(xValue, yValue)).toTensor[Float]
    require(output.nElement() == xValue.nElement())
    for (i <- 1 to xValue.dim()) {
      require((xValue.narrow(3, i, 1) - yValue).almostEqual(output.narrow(3, i, 1), 1e-8) == true)
    }
  }

  "4d mm" should "be able to generate correct result" in {
    val xValue = Tensor[Float]((2 until 50).toArray.map(_.toFloat), Array(2, 3, 2, 4))
    val yValue = Tensor[Float]((2 until 50).toArray.map(_.toFloat), Array(2, 3, 4, 2))
    val x = Variable[Float](inputShape = Shape(3, 2, 4))
    val y = Variable[Float](inputShape = Shape(3, 4, 2))
    val z = AutoGrad.mm(x, y)
    val model = Model[Float](Array(x, y), z)
    val output = model.forward(T(xValue, yValue)).toTensor[Float]
    val expect = Tensor[Float](Array[Float](80f, 94f,
      160f, 190f,
      608f, 654f,
      816f, 878f,
      1648f, 1726f,
      1984f, 2078f,
      3200f, 3310f,
      3664f, 3790f,
      5264f, 5406f,
      5856f, 6014f,
      7840f, 8014f,
      8560f, 8750f), Array(2, 3, 2, 2))
    require(output.almostEqual(expect, 1e-8) == true)

    val gradOutput = Tensor[Float]((20 until 44).toArray.map(_.toFloat), Array(2, 3, 2, 2))
    val gradInput = model.backward(T(xValue, yValue), gradOutput).toTable
    val expect2 = Tensor[Float](Array[Float](103f, 185f, 267f, 349f,
      113f, 203f, 293f, 383f,
      515f, 613f, 711f, 809f,
      557f, 663f, 769f, 875f,
      1055f, 1169f, 1283f, 1397f,
      1129f, 1251f, 1373f, 1495f,
      1723f, 1853f, 1983f, 2113f,
      1829f, 1967f, 2105f, 2243f,
      2519f, 2665f, 2811f, 2957f,
      2657f, 2811f, 2965f, 3119f,
      3443f, 3605f, 3767f, 3929f,
      3613f, 3783f, 3953f, 4123f), Array(2, 3, 2, 4))
    require(gradInput[Tensor[Float]](1).almostEqual(expect2, 1e-8) == true)
    val expect3 = Tensor[Float](Array[Float](172f, 180f,
      214f, 224f,
      256f, 268f,
      298f, 312f,
      604f, 628f,
      654f, 680f,
      704f, 732f,
      754f, 784f,
      1164f, 1204f,
      1222f, 1264f,
      1280f, 1324f,
      1338f, 1384f,
      1852f, 1908f,
      1918f, 1976f,
      1984f, 2044f,
      2050f, 2112f,
      2668f, 2740f,
      2742f, 2816f,
      2816f, 2892f,
      2890f, 2968f,
      3612f, 3700f,
      3694f, 3784f,
      3776f, 3868f,
      3858f, 3952f), Array(2, 3, 4, 2))
    require(gradInput[Tensor[Float]](2).almostEqual(expect3, 1e-8) == true)
  }

  "4d mm" should "be work with axes" in {
    val xValue = Tensor[Float](Array(2, 2, 3, 4)).rand()
    val yValue = Tensor[Float](Array(2, 2, 3, 4)).rand()
    val x = Variable[Float](inputShape = Shape(2, 3, 4))
    val y = Variable[Float](inputShape = Shape(2, 3, 4))
    val z = AutoGrad.mm(x, y, axes = List(3, 3))
    val model = Model[Float](Array(x, y), z)
    val input = T(xValue, yValue)
    val output = model.forward(input).toTensor[Float]
    model.backward(input, output)
  }

  "Constant variable" should "be work" in {
    val xValue = Tensor[Float](Array(2, 3, 4)).rand()
    val yValue = Tensor[Float](Array(2, 3, 4)).rand()
    val x = new Constant[Float](xValue)
    val y = Variable[Float](inputShape = Shape(3, 4))
    val z = y + x
    val model = Model[Float](y, z)
    val output = model.forward(yValue).toTensor[Float]
    require(output.almostEqual(xValue.add(yValue), 1e-8))
    model.backward(yValue, xValue).toTensor[Float]
  }
}

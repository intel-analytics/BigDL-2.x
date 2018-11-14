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
}

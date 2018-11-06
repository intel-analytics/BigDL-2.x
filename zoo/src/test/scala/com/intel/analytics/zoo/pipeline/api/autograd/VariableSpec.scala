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


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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalExpand
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class InternalExpandSpec extends KerasBaseSpec {
  "InternalExpand" should "generate correct output" in {
    val tgtSizes = Array(5, 4, 3)
    val layer = InternalExpand[Float](tgtSizes)
    val input = Tensor[Float](5, 4, 1).rand()
    val gradOutput = Tensor[Float](5, 4, 3).rand()
    val output = layer.forward(input)
    for (i <- 1 to 3) {
      require(output.narrow(3, i, 1).almostEqual(input, 1e-8) == true)
    }
    val gradInput = layer.backward(input, gradOutput)
    require(gradInput.nElement() == input.nElement())
  }

  "InternalExpand with expand batch dim" should "generate correct output" in {
    val tgtSizes = Array(5, 4, 3)
    val layer = InternalExpand[Float](tgtSizes, true)
    val input = Tensor[Float](1, 4, 3).rand()
    val gradOutput = Tensor[Float](5, 4, 3).rand()
    val output = layer.forward(input)
    for (i <- 1 to 5) {
      require(output.narrow(1, i, 1).almostEqual(input, 1e-8) == true)
    }
    val gradInput = layer.backward(input, gradOutput)
    require(gradInput.nElement() == input.nElement())
  }
}

class ExpandSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Expand[Float](tgtSizes = Array(3, 2, 4), inputShape = Shape(2, 1))
    layer.build(Shape(3, 2, 1))
    val input = Tensor[Float](3, 2, 1).rand()
    runSerializationTest(layer, input)
  }
}


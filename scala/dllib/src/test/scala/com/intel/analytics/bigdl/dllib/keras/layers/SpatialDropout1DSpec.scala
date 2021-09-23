/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.models.Sequential
import com.intel.analytics.bigdl.dllib.keras.serializer.ModuleSerializationTest


class SpatialDropout1DSpec extends KerasBaseSpec {

  "SpatialDropout1D forward and backward" should "work properly" in {
    val seq = Sequential[Float]()
    val layer = SpatialDropout1D[Float](0.5, inputShape = Shape(3, 4))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3, 4))
    val input = Tensor[Float](2, 3, 4).rand()
    val output = seq.forward(input)
    val gradInput = seq.backward(input, output)
  }

}

class SpatialDropout1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = SpatialDropout1D[Float](0.5, inputShape = Shape(3, 4))
    layer.build(Shape(2, 3, 4))
    val input = Tensor[Float](2, 3, 4).rand()
    runSerializationTest(layer, input)
  }
}

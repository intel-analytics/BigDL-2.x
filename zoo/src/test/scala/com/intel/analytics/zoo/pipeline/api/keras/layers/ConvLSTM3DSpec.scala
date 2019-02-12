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
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class ConvLSTM3DSpec extends ZooSpecHelper {

  "ConvLSTM3D" should "forward and backward properly with correct output shape" in {
    val layer = ConvLSTM3D[Float](10, 3, inputShape = Shape(5, 4, 8, 10, 12))
    layer.build(Shape(-1, 5, 4, 8, 10, 12))
    val input = Tensor[Float](Array(3, 5, 4, 8, 10, 12)).rand()
    val output = layer.forward(input)
    val expectedOutputShape = layer.getOutputShape().toSingle().toArray
    val actualOutputShape = output.size()
    require(expectedOutputShape.drop(1).sameElements(actualOutputShape.drop(1)))
    val gradInput = layer.backward(input, output)
  }
}

class ConvLSTM3DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ConvLSTM3D[Float](10, 3, inputShape = Shape(5, 4, 8, 10, 12))
    layer.build(Shape(2, 5, 4, 8, 10, 12))
    val input = Tensor[Float](2, 5, 4, 8, 10, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

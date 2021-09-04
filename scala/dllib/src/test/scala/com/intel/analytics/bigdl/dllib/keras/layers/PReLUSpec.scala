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

package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.nn.{PReLU => BPReLU}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{PReLU => ZPReLU}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class PReLUSpec extends ZooSpecHelper {

  "PReLU 3D Zoo" should "be the same as BigDL" in {
    val blayer = BPReLU[Float](2)
    val zlayer = ZPReLU[Float](2, inputShape = Shape(3, 4))
    zlayer.build(Shape(-1, 3, 4))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3, 4))
    val input = Tensor[Float](Array(2, 3, 4)).rand()
    compareOutputAndGradInputSetWeights(blayer, zlayer, input)
  }

  "PReLU 4D Zoo" should "be the same as BigDL" in {
    val blayer = BPReLU[Float]()
    val zlayer = ZPReLU[Float](inputShape = Shape(4, 8, 8))
    zlayer.build(Shape(-1, 4, 8, 8))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 4, 8, 8))
    val input = Tensor[Float](Array(3, 4, 8, 8)).rand()
    compareOutputAndGradInputSetWeights(blayer, zlayer, input)
  }

}

class PReLUSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = PReLU[Float](inputShape = Shape(4, 5))
    layer.build(Shape(2, 4, 5))
    val input = Tensor[Float](2, 4, 5).rand()
    runSerializationTest(layer, input)
  }
}

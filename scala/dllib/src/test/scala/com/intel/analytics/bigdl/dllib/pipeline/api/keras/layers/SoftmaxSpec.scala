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

package com.intel.analytics.bigdl.dllib.zooKeras.layers

import com.intel.analytics.bigdl.common.utils.Shape
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.zooKeras.serializer.ModuleSerializationTest

class SoftMaxSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Activation[Float]("softmax", inputShape = Shape(2, 2, 2))
    layer.build(Shape(3, 2, 2, 2))
    val input = Tensor[Float](3, 2, 2, 2).rand()
    runSerializationTest(layer, input)
  }
}

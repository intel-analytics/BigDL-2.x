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

package com.intel.analytics.zoo.pipeline.api.keras2.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Keras2Test, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

class Softmax extends KerasBaseSpec {

  "softmax with 4d input" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2, 2, 2])
        |input = np.random.random([2, 2, 2, 2])
        |output_tensor = Activation('softmax')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Activation[Float]("softmax", inputShape = Shape(2, 2, 2))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }
}

class SoftMaxSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Activation[Float]("softmax", inputShape = Shape(2, 2, 2))
    layer.build(Shape(3, 2, 2, 2))
    val input = Tensor[Float](3, 2, 2, 2).rand()
    runSerializationTest(layer, input)
  }
}


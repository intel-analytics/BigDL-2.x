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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.common.utils.Shape
import com.intel.analytics.bigdl.dllib.zooKeras.models.Sequential
import com.intel.analytics.bigdl.dllib.zooKeras.serializer.ModuleSerializationTest


class MaxPooling3DSpec extends KerasBaseSpec {

  "MaxPooling3D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 20, 15, 35])
        |input = np.random.random([2, 3, 20, 15, 35])
        |output_tensor = MaxPooling3D((2, 2, 3), dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = MaxPooling3D[Float](poolSize = (2, 2, 3), inputShape = Shape(3, 20, 15, 35))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3, 10, 7, 11))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class MaxPooling3DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = MaxPooling3D[Float](inputShape = Shape(3, 20, 15, 35))
    layer.build(Shape(2, 3, 20, 15, 35))
    val input = Tensor[Float](2, 3, 20, 15, 35).rand()
    runSerializationTest(layer, input)
  }
}

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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasBaseSpec
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.api.keras.layers.Keras2Test

import scala.util.Random

class MaxPooling1DSpec extends KerasBaseSpec {

  "MaxPooling1D valid mode" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 12])
        |input = np.random.random([3, 12, 12])
        |output_tensor = MaxPooling1D(pool_size=3)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = MaxPooling1D[Float](poolSize = 3, inputShape = Shape(12, 12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4, 12))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "MaxPooling1D same mode" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[20, 32])
        |input = np.random.random([3, 20, 32])
        |output_tensor = MaxPooling1D(strides=1, padding="same")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = MaxPooling1D[Float](strides = 1, padding = "same",
      inputShape = Shape(20, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 20, 32))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class MaxPooling1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = MaxPooling1D[Float](inputShape = Shape(12, 12))
    layer.build(Shape(2, 12, 12))
    val input = Tensor[Float](2, 12, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

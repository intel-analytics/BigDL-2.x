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
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Keras2Test, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class MaxPooling2DSpec extends KerasBaseSpec {
  "MaxPooling2D NCHW" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 24, 24])
        |input = np.random.random([2, 3, 24, 24])
        |output_tensor = MaxPooling2D(data_format="channels_first")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = MaxPooling2D[Float](inputShape = Shape(3, 24, 24))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "MaxPooling2D NHWC" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[32, 28, 5])
        |input = np.random.random([3, 32, 28, 5])
        |output_tensor = MaxPooling2D(pool_size=(2, 3), strides=(1, 2),
        |                            data_format="channels_last")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = MaxPooling2D[Float](poolSize = (2, 3), strides = (1, 2),
      dataFormat = "channels_last", inputShape = Shape(32, 28, 5))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "MaxPooling2D same Data Format" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 24, 24])
        |input = np.random.random([2, 3, 24, 24])
        |output_tensor = MaxPooling2D(strides=(1, 2), padding="same",
        |                             data_format="channels_first")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = MaxPooling2D[Float](strides = (1, 2), padding = "same",
      inputShape = Shape(3, 24, 24))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }
}

class MaxPooling2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = MaxPooling2D[Float](inputShape = Shape(3, 24, 24))
    layer.build(Shape(2, 3, 24, 24))
    val input = Tensor[Float](2, 3, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

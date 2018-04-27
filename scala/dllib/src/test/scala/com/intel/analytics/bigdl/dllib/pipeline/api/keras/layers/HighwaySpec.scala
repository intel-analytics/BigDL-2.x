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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class HighwaySpec extends KerasBaseSpec {

  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    in.length match {
      case 2 => Array(in(1).t(), in(0).t()) // without bias
      case _ => Array(in(1).t(), in(3), in(0).t(), in(2)) // with bias
    }
  }

  "Highway computeOutputShape" should "work properly" in {
    val seq = Sequential[Float]()
    val klayer = Highway[Float](inputShape = Shape(6))
    seq.add(klayer)
    seq.add(Dense(5))
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 5))
  }

  "Highway" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[10])
        |input = np.random.random([4, 10])
        |output_tensor = Highway(activation="relu")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Highway[Float](inputShape = Shape(10), activation = "relu")
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 10))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "Highway without bias" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4])
        |input = np.random.random([2, 4])
        |output_tensor = Highway(activation="tanh", bias=False)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Highway[Float](activation = "tanh", bias = false, inputShape = Shape(4))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class HighwaySerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Highway[Float](activation = "tanh", bias = false, inputShape = Shape(4))
    layer.build(Shape(3, 4))
    val input = Tensor[Float](3, 4).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

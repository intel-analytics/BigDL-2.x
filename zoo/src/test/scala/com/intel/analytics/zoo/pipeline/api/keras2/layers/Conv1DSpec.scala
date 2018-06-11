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

class Conv1DSpec extends KerasBaseSpec {

  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    in.length match {
      case 1 => in
      case _ => Array(in(0).resize(Array(1) ++ in(0).size()), in(1))
    }
  }

  "Conv1D" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 20])
        |input = np.random.random([2, 12, 20])
        |output_tensor = Conv1D(64, 3)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Conv1D[Float](64, 3, inputShape = Shape(12, 20))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be(Array(-1, 10, 64))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "Conv1D without bias" should "be the same as Keras" taggedAs(Keras2Test) in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[20, 32])
        |input = np.random.random([2, 20, 32])
        |output_tensor = Conv1D(32, 4, activation="relu", use_bias=False,
        |                              strides=2)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Conv1D[Float](32, 4, activation = "relu", strides = 2,
      useBias = false, inputShape = Shape(20, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be(Array(-1, 9, 32))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class Conv1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Conv1D[Float](64, 3, inputShape = Shape(12, 20))
    layer.build(Shape(2, 12, 20))
    val input = Tensor[Float](2, 12, 20).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}

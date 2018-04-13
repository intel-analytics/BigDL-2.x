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

class DenseSpec extends KerasBaseSpec {

  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = Array(in(0).t(), in(1))

  "Dense" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |input = np.random.uniform(0, 1, [1, 3])
        |output_tensor = Dense(2, activation="relu")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val input = InputLayer[Float](inputShape = Shape(3), name = "input1")
    seq.add(input)
    val dense = Dense[Float](2, activation = "relu")
    seq.add(dense)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 2))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "Dense nD input" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[10, 5, 7])
        |input = np.random.uniform(0, 1, [2, 10, 5, 7])
        |output_tensor = \
        |Dense(2, init='one', input_shape=(10, 5, 7))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val input = Input[Float](inputShape = Shape(10, 5, 7))
    val dense = Dense[Float](2, init = "one").inputs(input)
    val model = Model(input, dense)
    model.getOutputShape().toSingle().toArray should be (Array(-1, 10, 5, 2))
    checkOutputAndGrad(model.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, precision = 1e-4)
  }

}

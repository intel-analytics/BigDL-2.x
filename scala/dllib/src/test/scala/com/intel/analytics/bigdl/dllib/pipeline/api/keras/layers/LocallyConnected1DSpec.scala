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

class LocallyConnected1DSpec extends KerasBaseSpec {

  def weightConverter(data: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    val out = new Array[Tensor[Float]](data.length)
    val d1l: Int = data(0).size(1)
    val d2l: Int = data(0).size(2)
    val d3l: Int = data(0).size(3)
    out(0) = Tensor(d1l, d3l, d2l)
    val page: Int = d2l * d3l
    for (i <- 0 until d1l * d2l * d3l) {
      val d1 = i / page + 1
      val d2 = (i % page) / d3l + 1
      val d3 = (i % page) % d3l + 1
      val v = data(0).valueAt(d1, d2, d3)
      out(0).setValue(d1, d3, d2, v)
    }
    if (data.length > 1) {
      out(1) = data(1)
    }
    out
  }

  "LocallyConnected1D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 24])
        |input = np.random.random([3, 12, 24])
        |output_tensor = LocallyConnected1D(32, 3, activation="relu")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = LocallyConnected1D[Float](32, 3, activation = "relu",
      inputShape = Shape(12, 24))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "LocallyConnected1D without bias" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[32, 32])
        |input = np.random.random([2, 32, 32])
        |output_tensor = LocallyConnected1D(64, 4, subsample_length=2,
        |                                   bias=False)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = LocallyConnected1D[Float](64, 4, subsampleLength = 2,
      bias = false, inputShape = Shape(32, 32))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

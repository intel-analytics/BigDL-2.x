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

class Convolution2DSpec extends KerasBaseSpec {
  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    in.length match {
      case 1 => in
      case _ => Array(in(0).resize(Array(1) ++ in(0).size()), in(1))
    }
  }

  "Convolution2D NCHW" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 24, 24])
        |input = np.random.random([2, 3, 24, 24])
        |output_tensor = Convolution2D(64, 2, 5, activation="relu",
        |                              dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Convolution2D[Float](64, 2, 5, activation = "relu", inputShape = Shape(3, 24, 24))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, 1e-3)
  }

  "Convolution2D NHWC" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[24, 24, 3])
        |input = np.random.random([2, 24, 24, 3])
        |output_tensor = Convolution2D(32, 4, 6, border_mode="same",
        |                              dim_ordering="tf")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Convolution2D[Float](32, 4, 6, dimOrdering = "tf",
      borderMode = "same", inputShape = Shape(24, 24, 3))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, 1e-3)
  }

  "Conv2D without bias" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 24, 24])
        |input = np.random.random([2, 3, 24, 24])
        |output_tensor = Convolution2D(64, 2, 5, bias=False, subsample=(2, 3),
        |                              init="normal", dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Conv2D[Float](64, 2, 5, subsample = (2, 3),
      init = "normal", bias = false, inputShape = Shape(3, 24, 24))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, 1e-4)
  }
}

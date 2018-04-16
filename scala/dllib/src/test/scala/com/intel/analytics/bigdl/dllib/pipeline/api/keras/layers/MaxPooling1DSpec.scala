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

class MaxPooling1DSpec extends KerasBaseSpec {

  "MaxPooling1D valid mode" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 12])
        |input = np.random.random([3, 12, 12])
        |output_tensor = MaxPooling1D(pool_length=3)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = MaxPooling1D[Float](poolLength = 3, inputShape = Shape(12, 12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4, 12))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "MaxPooling1D same mode" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[20, 32])
        |input = np.random.random([3, 20, 32])
        |output_tensor = MaxPooling1D(stride=1, border_mode="same")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = MaxPooling1D[Float](stride = 1, borderMode = "same",
      inputShape = Shape(20, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 20, 32))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

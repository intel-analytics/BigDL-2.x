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

package com.intel.analytics.zoo.models.seq2seq

import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers._

import scala.reflect.ClassTag

object Bridge {
  /**
   * [[Bridge]] defines how to transform encoder to decoder
   * @param bridgeType currently only support "dense | densenonlinear"
   * @param rnnType rnn type used for encoder and decoder
   * @param numLayers number of layers used in encoder and decoder
   * @param decoderHiddenSize hidden size of decoder
   * @param inputShape shape of input
   */
  def apply[@specialized(Float, Double) T: ClassTag](bridgeType: String,
    rnnType: String,
    numLayers: Int,
    decoderHiddenSize: Int,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]):
    KerasLayer[Tensor[T], Tensor[T], T] = {
    val numStates = if (rnnType.toLowerCase == "lstm") 2 * numLayers else numLayers
    val bridge = bridgeType.toLowerCase() match {
      case "dense" =>
        Dense(decoderHiddenSize * numStates, bias = false, inputShape = inputShape)
      case "densenonlinear" =>
        Dense(decoderHiddenSize * numStates, activation = "tanh", bias = false,
          inputShape = inputShape)
      case _ => throw new IllegalArgumentException(s"Please use " +
        s"Bridge(bridge: KerasLayer[Tensor[T], Tensor[T], T], inputShape: Shape)" +
        s"to create a bridge")
    }
    bridge
  }
}

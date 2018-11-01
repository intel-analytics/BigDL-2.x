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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

import scala.reflect.ClassTag

/**
 * [[Bridge]] defines how to transform encoder to decoder
 * @param bridge keras layer used to transform encoder state
 * @param inputShape shape of input
 */
class Bridge[T: ClassTag](rnnType: String,
  bridge: KerasLayer[Tensor[T], Tensor[T], T],
  inputShape: Shape)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))
    with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = Sequential()
    layer.add(InputLayer(KerasUtils.removeBatch(inputShape)))
    layer.add(bridge)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Bridge {
  /**
   * [[Bridge]] defines how to transform encoder to decoder
   * @param bridge keras layer used to transform encoder state
   * @param inputShape shape of input
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    rnnType: String,
    bridge: KerasLayer[Tensor[T], Tensor[T], T],
    inputShape: Shape)(implicit ev: TensorNumeric[T]): Bridge[T] = {
    new Bridge[T](rnnType, bridge, inputShape)
  }

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
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Bridge[T] = {
    val numStates = if (rnnType.toLowerCase == "lstm") 2 * numLayers else numLayers
    val bridge = bridgeType.toLowerCase() match {
      case "dense" =>
        Dense(decoderHiddenSize * numStates, bias = false)
      case "densenonlinear" =>
        Dense(decoderHiddenSize * numStates, activation = "tanh", bias = false)
      case _ => throw new IllegalArgumentException(s"Please use " +
        s"Bridge(bridge: KerasLayer[Tensor[T], Tensor[T], T], inputShape: Shape)" +
        s"to create a bridge")
    }
    Bridge[T](rnnType, bridge, inputShape)
  }
}

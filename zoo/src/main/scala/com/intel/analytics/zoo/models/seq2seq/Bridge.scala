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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, SingleShape, T}
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
class Bridge[T: ClassTag](
  bridge: KerasLayer[Tensor[T], Tensor[T], T],
  inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](KerasUtils.addBatch(inputShape))
    with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = {
    val layerNum = inputShape.toMulti().size
    val stateNum = if (inputShape.toMulti().head.isInstanceOf[SingleShape]) 1
    else 2

    val layer = Sequential()
    if (stateNum == 2 || layerNum != 1) {
      layer.add(new KerasLayerWrapper[T](new InternalJoinTable(2, -1)
        .asInstanceOf[AbstractModule[Activity, Activity, T]], KerasUtils.removeBatch(inputShape)))
    }

    layer.add(bridge)

    if (layerNum != 1 || stateNum == 2) {
      if (stateNum == 2) {
        layer.add(new KerasLayerWrapper[T](new InternalSplitTensor[T](2, layerNum, true)
          .asInstanceOf[AbstractModule[Activity, Activity, T]]))
      } else {
        layer.add(new KerasLayerWrapper[T](new InternalSplitTensor[T](2, layerNum, false)
          .asInstanceOf[AbstractModule[Activity, Activity, T]]))
      }
    }

    layer.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object Bridge {
  /**
   * [[Bridge]] defines how to transform encoder to decoder
   * @param bridgeType currently only support "dense | densenonlinear"
   * @param rnnType style of recurrent unit, one of [SimpleRNN, LSTM, GRU]
   * @param numLayers number of layers used in encoder and decoder
   * @param decoderHiddenSize hidden size of decoder
   * @param inputShape shape of input
   */
  def apply[@specialized(Float, Double) T: ClassTag](bridgeType: String,
    rnnType: String,
    numLayers: Int,
    decoderHiddenSize: Int,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]):
    Bridge[T] = {
    val rnnName = rnnType.toLowerCase
    require(rnnName == "lstm" || rnnName == "gru" || rnnName == "simplernn",
      "rnnType has to be lstm | gru | simplernn")
    val numStates = if (rnnName == "lstm") 2 * numLayers else numLayers
    val bridge = bridgeType.toLowerCase() match {
      case "dense" =>
        Dense(decoderHiddenSize * numStates, bias = false, inputShape = inputShape)
      case "densenonlinear" =>
        Dense(decoderHiddenSize * numStates, activation = "tanh", bias = false,
          inputShape = inputShape)
      case _ => throw new IllegalArgumentException(s"Please use " +
        s"Bridge(rnnType: String, bridge: KerasLayer[Tensor[T], Tensor[T], T]," +
        s"inputShape: Shape) to create a bridge")
    }
    new Bridge(bridge, inputShape)
  }
}

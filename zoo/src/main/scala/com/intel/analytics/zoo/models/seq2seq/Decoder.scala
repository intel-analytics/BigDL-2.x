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
import com.intel.analytics.bigdl.utils._

import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.SelectTable
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.{Sequential}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * [[Decoder]] A generic recurrent neural network decoder
 * @param rnns rnn layers used for decoder, support stacked rnn layers
 * @param embedding embedding layer in decoder
 * @param inputShape shape of input
 */
class Decoder[T: ClassTag](val rnns: Array[Recurrent[T]],
  val embedding: KerasLayer[Tensor[T], Tensor[T], T],
  val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Tensor[T], T](KerasUtils.addBatch(inputShape))
    with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Tensor[T], T] = {
    val layer = Sequential()
    // get decoder input
    layer.add(SelectTable(1, KerasUtils.removeBatch(inputShape)))
    if (embedding != null) layer.add(embedding)
    rnns.foreach(layer.add(_))
    layer.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    rnns.last.getOutputShape()
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    val states = input.toTable[Tensor[T]](2)

    // split states from numLayers x batch x hidden to Array(batch, hidden)
    val splitStates = if (rnns.head.getName().toLowerCase.contains("lstm"))
      Utils.splitToTable(states, rnns.size)
    else Utils.splitToTensor(states, rnns.size)

    for ((rnn, state) <- rnns.zip(splitStates)) {
      rnn.setHiddenState(state)
    }
    output = labor.updateOutput(input)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    val rnnsGradInput = labor.updateGradInput(input, gradOutput)
    val gradStates = rnns.map(_.getGradHiddenState())

    // concat states from Array(batch x hidden) to numLayers x batch x hidden
    val catGradstates = Utils.join(gradStates)
    gradInput = T(rnnsGradInput, catGradstates)
    gradInput
  }
}

object Decoder {
  /**
   * [[Decoder]] A generic recurrent neural network decoder
   * @param rnns rnn layers used for decoder, support stacked rnn layers
   * @param embedding embedding layer in decoder
   * @param inputShape shape of input
   */
  def apply[@specialized(Float, Double) T: ClassTag](rnns: Array[Recurrent[T]],
    embedding: KerasLayer[Tensor[T], Tensor[T], T],
    inputShape: Shape)(implicit ev: TensorNumeric[T]): Decoder[T] = {
    new Decoder[T](rnns, embedding, inputShape)
  }

  /**
   * [[Decoder]] A generic recurrent neural network decoder
   * @param rnnType rnn type used for decoder, currently only support "lstm | gru"
   * @param numLayers number of layers used in decoder
   * @param hiddenSize hidden size of decoder
   * @param embedding embedding layer in decoder
   * @param inputShape shape of input
   */
  def apply[@specialized(Float, Double) T: ClassTag](rnnType: String,
    numLayers: Int,
    hiddenSize: Int,
    embedding: KerasLayer[Tensor[T], Tensor[T], T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Decoder[T] = {
    val rnn = new ArrayBuffer[Recurrent[T]]()
    rnnType.toLowerCase() match {
      case "lstm" =>
        for (i <- 1 to numLayers) rnn.append(LSTM(hiddenSize, returnSequences = true))
      case "gru" => {
        for (i <- 1 to numLayers) rnn.append(GRU(hiddenSize, returnSequences = true))
      }
      case _ => throw new IllegalArgumentException(s"Please use " +
        s"Decoder(rnn: Array[Recurrent[T]], embedding: KerasLayer[Activity, Activity, T])" +
        s"to create a decoder")
    }
    Decoder[T](rnn.toArray, embedding, inputShape)
  }
}

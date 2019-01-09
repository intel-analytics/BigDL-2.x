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
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * [[RNNEncoder]] A generic recurrent neural network encoder
 *
 * @param rnns rnn layers used for encoder, support stacked rnn layers
 * @param embedding embedding layer in encoder
 * @param inputShape shape of input
 */
class RNNEncoder[T: ClassTag](val rnns: Array[Recurrent[T]],
  val embedding: KerasLayer[Tensor[T], Tensor[T], T],
  val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Encoder[T](inputShape) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Activity, T] = {
    val layer = Sequential()
    layer.add(InputLayer(KerasUtils.removeBatch(inputShape)))
    if (embedding != null) {
      layer.add(embedding)
    }
    rnns.foreach(layer.add(_))
    layer.asInstanceOf[AbstractModule[Tensor[T], Activity, T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    // batch x seq x hidden
    val rnnShape = labor.getOutputShape()
    val sizes = rnnShape.toSingle().toArray

    val statesShape = rnns.map { rnn =>
      MultiShape(rnn.getHiddenShape().map(x => Shape(Array(sizes(0), x) ++ sizes.drop(3))).toList)
    }

    Shape(List(rnnShape, MultiShape(statesShape.toList)))
  }

  // ouput is T(rnnOutput, T(layer1states, layer2states, ...))
  override def updateOutput(input: Tensor[T]): Activity = {
    val laborOutput = labor.updateOutput(input)
    val states = rnns.map(_.getHiddenState())

    output = T(laborOutput, T.array(states))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Activity): Tensor[T] = {
    val rnnGradOutput = gradOutput.toTable[Tensor[T]](1)
    val gradStates = gradOutput.toTable[Table](2)

    var i = 0
    while (i < rnns.size) {
      rnns(i).setGradHiddenState(gradStates(i + 1))
      i += 1
    }
    gradInput = labor.updateGradInput(input, rnnGradOutput)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Activity): Unit = {
    val rnnGradOutput = gradOutput.toTable[Tensor[T]](1)
    val gradStates = gradOutput.toTable[Table](2)

    var i = 0
    while (i < rnns.size) {
      rnns(i).setGradHiddenState(gradStates(i + 1))
      i += 1
    }
    labor.accGradParameters(input, rnnGradOutput)
  }
}

object RNNEncoder {
  /**
   * [[RNNEncoder]] A generic recurrent neural network encoder
   *
   * @param rnns rnn layers used for encoder, support stacked rnn layers
   * @param embedding embedding layer in encoder
   * @param inputShape shape of input
   */
  def apply[@specialized(Float, Double) T: ClassTag](rnns: Array[Recurrent[T]],
    embedding: KerasLayer[Tensor[T], Tensor[T], T],
    inputShape: Shape)(implicit ev: TensorNumeric[T]): RNNEncoder[T] = {
    new RNNEncoder[T](rnns, embedding, inputShape)
  }

  /**
   * [[RNNEncoder]] A generic recurrent neural network encoder
   *
   * @param rnnType style of recurrent unit, one of [SimpleRNN, LSTM, GRU]
   * @param numLayers number of layers used in encoder
   * @param hiddenSize hidden size of encoder
   * @param embedding embedding layer in encoder
   * @param inputShape shape of input
   */
  def apply[@specialized(Float, Double) T: ClassTag](rnnType: String,
    numLayers: Int,
    hiddenSize: Int,
    embedding: KerasLayer[Tensor[T], Tensor[T], T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): RNNEncoder[T] = {
    val rnn = new ArrayBuffer[Recurrent[T]]()
    rnnType.toLowerCase() match {
      case "lstm" =>
        for (i <- 1 to numLayers) rnn.append(LSTM(hiddenSize, returnSequences = true))
      case "gru" =>
        for (i <- 1 to numLayers) rnn.append(GRU(hiddenSize, returnSequences = true))
      case "simplernn" =>
        for (i <- 1 to numLayers) rnn.append(SimpleRNN(hiddenSize, returnSequences = true))
      case _ => throw new IllegalArgumentException(s"Please use " +
        s"RNNEncoder(rnn: Array[Recurrent[T]], embedding: KerasLayer[Activity, Activity, T])" +
        s"to create a encoder")
    }
    RNNEncoder[T](rnn.toArray, embedding, inputShape)
  }
}

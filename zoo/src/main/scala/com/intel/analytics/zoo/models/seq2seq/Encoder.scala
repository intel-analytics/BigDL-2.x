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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T, Table}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers.{GRU, LSTM, Recurrent}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class Encoder[T: ClassTag](val rnn: Array[Recurrent[T]],
  val embedding: KerasLayer[Tensor[T], Tensor[T], T],
  val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Table, T](KerasUtils.addBatch(inputShape))
    with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Table, T] = {
    val layer = Sequential()
    if (embedding != null) layer.add(embedding)
    rnn.foreach(layer.add(_))
    layer.asInstanceOf[AbstractModule[Tensor[T], Table, T]]
  }

  override def updateOutput(input: Tensor[T]): Table = {
    val laborOutput = labor.updateOutput(input)

    // concat array of states(batch x hidden) to (batch x hidden x numLayers)
    val states = rnn.map(_.getHiddenState())
    val headState = states.head
    val catStates = if (headState.isTensor) {
      // non LSTM
      Tensor(states.map(_.toTensor.storage().array()).flatten,
        shape = headState.toTensor.size() ++ Array(rnn.length))
    } else {
      // LSMT
      T(Tensor(states.map(_.toTable[Tensor[T]](0).storage().array()).flatten,
        shape = headState.toTensor.size() ++ Array(rnn.length)),
      Tensor(states.map(_.toTable[Tensor[T]](1).storage().array()).flatten,
        shape = headState.toTensor.size() ++ Array(rnn.length)))
      }
    output = T(laborOutput, catStates)
    output
  }
}

object Encoder {
  def apply[@specialized(Float, Double) T: ClassTag](rnn: Array[Recurrent[T]],
    embedding: KerasLayer[Tensor[T], Tensor[T], T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Encoder[T] = {
    new Encoder[T](rnn, embedding, inputShape)
  }

  def apply[@specialized(Float, Double) T: ClassTag](rnnType: String,
    numLayers: Int,
    hiddenSize: Int,
    dropout: Double,
    embedding: KerasLayer[Tensor[T], Tensor[T], T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Encoder[T] = {
    val rnn = new ArrayBuffer[Recurrent[T]]()
    rnnType.toLowerCase() match {
      case "lstm" =>
        for (i <- 1 to numLayers) rnn.append(LSTM(hiddenSize, returnSequences = true))
      case "gru" => {
        for (i <- 1 to numLayers) rnn.append(GRU(hiddenSize, returnSequences = true))
      }
      case _ => throw new IllegalArgumentException(s"Please use " +
        s"Encoder(rnn: Array[Recurrent[T]], embedding: KerasLayer[Activity, Activity, T])" +
        s"to create a encoder")
    }
    Encoder[T](rnn.toArray, embedding, inputShape)
  }
}
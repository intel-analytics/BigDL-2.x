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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn._

import scala.reflect.ClassTag
import com.intel.analytics.zoo.models.common.ZooModel

class Seq2seq[T: ClassTag](val encoderCells: Array[Cell[T]],
                           val decoderCells: Array[Cell[T]],
                           val preEncoder: AbstractModule[Activity, Activity, T] = null,
                           val preDecoder: AbstractModule[Activity, Activity, T] = null,
                           val bridges: Bridge = new PassThroughBridge())
  (implicit ev: TensorNumeric[T]) extends ZooModel[Activity, Activity, T] {
  var preDecoderInput: Tensor[T] = null
  var decoderInput: Tensor[T] = null
  var encoderInput: Tensor[T] = null
  var encoderOutput: Tensor[T] = null
  var encoder: AbstractModule[Activity, Activity, T] = null
  var decoder: AbstractModule[Activity, Activity, T] = null
  private var enc: Array[ZooRecurrent[T]] = null
  private var dec: Array[Recurrent[T]] = null

  private var loopPreOutput: Boolean = false
  private var seqLen: Int = 0
  def setLoopPreOutput(seqLen: Int) = {
    loopPreOutput = true
    this.seqLen = seqLen
    //TODO: NEED TO VERIFY IF THE PARAMTERS ARE THE SAME AFTER SET
    modules.clear()
    modules += buildModel()
  }
  def clearLoopPreOutput() = {
    loopPreOutput = false
    this.seqLen = 0
    modules.clear()
    modules += buildModel()
  }

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    encoder = buildEncoder()
    val model = Sequential[T]().add(encoder)
    if (bridges.isInstanceOf[InitialStateBridge[T]]) {
      bridges.asInstanceOf[InitialStateBridge[T]].activations.foreach(activation =>
        if (activation != null) activation.foreach(module =>
          model.add(module)))
    }
    decoder = buildDecoder()
    model.add(decoder)
    model
  }

  private def buildEncoder(): AbstractModule[Activity, Activity, T] = {
    val model = Sequential[T]()
    if (preEncoder != null) model.add(preEncoder)
    enc = encoderCells.map {cell =>
      val rec = new ZooRecurrent().add(cell)
      model.add(rec)
      rec
    }
    model
  }

  private def buildDecoder(): AbstractModule[Activity, Activity, T] = {
    val model = Sequential[T]()
    if (preDecoder != null) model.add(preDecoder)
    dec = if (loopPreOutput) {
      require(seqLen > 0, "Please setLoopPreOutput if you want to use i-1th output as i th input")
      val dec = new ZooRecurrentDecoder(seqLen).add(MultiRNNCell(decoderCells))
      model.add(dec)
      Array(dec)
    } else {
      decoderCells.map {cell =>
        val dec = new ZooRecurrent().add(cell)
        model.add(dec)
        dec
      }
    }
    model
  }

  override def updateOutput(input: Activity): Activity = {
    if (loopPreOutput) {
      encoderInput = input.toTensor
      preDecoderInput = input.toTensor.select(2, input.toTensor.size(2))
    } else {
      encoderInput = input.toTable(1)
      preDecoderInput = input.toTable(2)
    }
    encoderOutput = encoder.forward(encoderInput).toTensor
    bridges.forwardStates(enc, dec)

    decoderInput = if (preDecoder != null) preDecoder.forward(preDecoderInput).toTensor
      else preDecoderInput
    output = decoder.forward(decoderInput)
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    val decoderGradInput = decoder.backward(decoderInput, gradOutput).toTensor
    if (preDecoder != null) {
      if (loopPreOutput) {
        preDecoder.backward(preDecoderInput, decoderGradInput.select(2, 1).contiguous())
      } else preDecoder.backward(preDecoderInput, decoderGradInput)
    }
    bridges.backwardStates(enc, dec)
    gradInput = encoder.backward(encoderInput, Tensor[T](encoderOutput.size())).toTensor

    gradInput.toTensor
  }

  override def clearState() : this.type = {
    super.clearState()
    preDecoderInput = null
    decoderInput = null
    encoderInput = null
    encoderOutput = null
    model.clearState()
    this
  }

  override def reset(): Unit = model.reset()
}

object Seq2seq {
  def apply[@specialized(Float, Double) T: ClassTag](encoderCells: Array[Cell[T]],
     decoderCells: Array[Cell[T]], preEncoder: AbstractModule[Activity, Activity, T] = null,
     preDecoder: AbstractModule[Activity, Activity, T] = null,
     bridges: Bridge = new PassThroughBridge())
    (implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    new Seq2seq[T](encoderCells, decoderCells, preEncoder, preDecoder, bridges)
  }
}

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
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag
import com.intel.analytics.zoo.models.common.ZooModel

/**
 * [[Seq2seq]] Sequence-to-sequence recurrent neural networks which maps input
 * sequences to output sequences.
 * @param encoderCells List of cells for encoder
 * @param decoderCells List of cells for decoder
 * @param preEncoder Before feeding input to encoder, pass it through preEncoder
 * @param preDecoder Before feeding input to decoder, pass it through preDecoder
 * @param bridges Bridges used to pass states between encoder and decoder.
 *                Default is PassThroughBridge
 */
class Seq2seq[T: ClassTag](encoderCells: Array[Cell[T]],
                           decoderCells: Array[Cell[T]],
                           preEncoder: AbstractModule[Activity, Activity, T] = null,
                           preDecoder: AbstractModule[Activity, Activity, T] = null,
                           bridges: Bridge = new PassThroughBridge(),
                           maskZero: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends ZooModel[Activity, Tensor[T], T] {
  private var preDecoderInput: Tensor[T] = null
  private var decoderInput: Tensor[T] = null
  private var encoderInput: Tensor[T] = null
  private var encoderOutput: Tensor[T] = null
  private var enc: Array[ZooRecurrent[T]] = null
  private var dec: Array[Recurrent[T]] = null

  var encoder: Sequential[T] = null
  var decoder: Sequential[T] = null

  private var seqLen: Int = 0
  private var stopSign: Tensor[T] = null
  private var loopFunc: (Tensor[T]) => (Tensor[T]) = null

  /**
   * defines prediction in last time step will be used as input in next time step in decoder
   * @param maxLen max sequence length of output
   * @param stopSign if prediction is the same with stopSign, it will stop predict
   * @param func before feeding output at last time step to next time step,
   *             pass it through loopFunc
   * @return this container
   */
  def setLoop(maxLen: Int, stopSign: Tensor[T] = null,
              func: (Tensor[T]) => (Tensor[T]) = null): Unit = {
    seqLen = maxLen
    this.stopSign = stopSign
    loopFunc = func
    modules.clear()
    modules += buildModel()
  }

  /**
   * clear setLoop settings
   */
  def clearLoop(): Unit = {
    seqLen = 0
    this.stopSign = null
    loopFunc = null
    modules.clear()
    modules += buildModel()
  }

  override def buildModel(): AbstractModule[Activity, Tensor[T], T] = {
    encoder = buildEncoder()
    val model = Sequential[T]().add(encoder)
    if (preDecoder != null) model.add(preDecoder)
    decoder = buildDecoder()
    model.add(decoder)
    model.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }

  private def buildEncoder(): Sequential[T] = {
    val model = Sequential[T]()
    if (preEncoder != null) model.add(preEncoder)
    enc = encoderCells.map {cell =>
      val rec = new ZooRecurrent(maskZero = maskZero).add(cell)
      model.add(rec)
      rec
    }
    model
  }

  private def buildDecoder(): Sequential[T] = {
    val model = Sequential[T]()
    dec = if (seqLen != 0 || loopFunc != null || stopSign != null) {
      require(seqLen > 0, "SeqLen needs be great than 0. Please use setLoopPreOutput to set seqLen")
      val cells = if (decoderCells.length == 1) decoderCells.head else MultiRNNCell(decoderCells)
      val recDec = new ZooRecurrentDecoder(seqLen, stopSign, loopFunc).add(cells)
      model.add(recDec)
      Array(recDec)
    } else {
      decoderCells.map {cell =>
        val rec = new ZooRecurrent(maskZero = maskZero).add(cell)
        model.add(rec)
        rec
      }
    }
    model
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    encoderInput = input.toTable(1)
    preDecoderInput = input.toTable(2)

    encoderOutput = encoder.forward(encoderInput).toTensor

    decoderInput = if (preDecoder != null) preDecoder.forward(preDecoderInput).toTensor
      else preDecoderInput

    if (bridges != null) bridges.forwardStates(enc, dec)
    output = decoder.forward(decoderInput).toTensor
    output
  }

  override def backward(input: Activity, gradOutput: Tensor[T]): Tensor[T] = {
    val decoderGradInput = decoder.backward(decoderInput, gradOutput).toTensor
    if (preDecoder != null) {
      if (preDecoderInput.dim < encoderInput.dim) {
        preDecoder.backward(preDecoderInput,
          decoderGradInput.select(Seq2seq.timeDim, 1).contiguous())
      } else preDecoder.backward(preDecoderInput, decoderGradInput)
    }

    if (bridges != null) bridges.backwardStates(enc, dec)
    gradInput = encoder.backward(encoderInput, Tensor[T](encoderOutput.size())).toTensor

    gradInput.toTensor
  }

  override def getParametersTable(): Table = {
    val pt = T()
    (encoderCells ++ decoderCells).foreach(cell => {
      val params = cell.getParametersTable()
      if (params != null) {
        params.keySet.foreach { key =>
          if (pt.contains(key)) {
            pt(key + Integer.toHexString(java.util.UUID.randomUUID().hashCode())) =
              params(key)
          } else {
            pt(key) = params(key)
          }
        }
      }
    })
    if (preEncoder != null) pt.add(preEncoder.getParametersTable())
    if (preDecoder != null) pt.add(preDecoder.getParametersTable())
    pt.add(bridges.toModel.getParametersTable())
    pt
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val params = model.parameters()

    val bridgeParam = bridges.toModel.parameters()
    (params._1 ++ bridgeParam._1, params._2 ++ bridgeParam._2)
  }

  override def clearState() : this.type = {
    super.clearState()
    preDecoderInput = null
    decoderInput = null
    encoderInput = null
    encoderOutput = null
    model.clearState()
    bridges.toModel.clearState()
    this
  }

  override def reset(): Unit = {
    model.reset()
    bridges.toModel.reset()
  }
}

object Seq2seq {
  def apply[@specialized(Float, Double) T: ClassTag](encoderCells: Array[Cell[T]],
     decoderCells: Array[Cell[T]], preEncoder: AbstractModule[Activity, Activity, T] = null,
     preDecoder: AbstractModule[Activity, Activity, T] = null,
     bridges: Bridge = new PassThroughBridge(),
     maskZero: Boolean = false)
    (implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    new Seq2seq[T](encoderCells, decoderCells, preEncoder, preDecoder, bridges, maskZero).build()
  }

  val timeDim = 2
}

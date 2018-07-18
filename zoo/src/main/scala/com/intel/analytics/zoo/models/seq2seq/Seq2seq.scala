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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
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
                           maskZero: Boolean = false,
                           generator: AbstractModule[Activity, Activity, T] = null)
  (implicit ev: TensorNumeric[T]) extends ZooModel[Activity, Tensor[T], T] {
  private var preDecoderInput: Tensor[T] = null
  private var decoderInput: Tensor[T] = null
  private var encoderInput: Tensor[T] = null
  private var encoderOutput: Tensor[T] = null
  private var decoderOutput: Tensor[T] = null
  private var enc: Array[ZooRecurrent[T]] = null
  private var dec: Array[Recurrent[T]] = null

  var encoder: Sequential[T] = null
  var decoder: Sequential[T] = null

  override def buildModel(): AbstractModule[Activity, Tensor[T], T] = {
    encoder = buildEncoder()
    val model = Sequential[T]().add(encoder)
    if (preDecoder != null) model.add(preDecoder)
    decoder = buildDecoder()
    model.add(decoder)
    if (generator != null) model.add(generator)
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
    dec = decoderCells.map {cell =>
        val rec = new ZooRecurrent(maskZero = maskZero).add(cell)
        model.add(rec)
        rec
      }
    model
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    encoderInput = input.toTable(1)
    preDecoderInput = input.toTable(2)

    encoderOutput = encoder.forward(encoderInput).toTensor
    decoderInput = if (preDecoder != null) {
      preDecoder.forward(preDecoderInput).toTensor
    } else preDecoderInput

    if (bridges != null) bridges.forwardStates(enc, dec)
    decoderOutput = decoder.forward(decoderInput).toTensor

    output = if (generator != null) generator.forward(decoderOutput).toTensor
    else decoderOutput
    output
  }

  override def backward(input: Activity, gradOutput: Tensor[T]): Tensor[T] = {
    val decoderGradoutput = if (generator != null) generator.backward(decoderOutput, gradOutput)
    else gradOutput
    val decoderGradInput = decoder.backward(decoderInput, decoderGradoutput).toTensor
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

  def inference(input: Table, maxSeqLen: Int = 30, stopSign: Tensor[T] = null,
                infer: TensorModule[T] = null): Tensor[T] = {
    val sent1 = input.toTable[Tensor[T]](1)
    val sent2 = input.toTable[Tensor[T]](2)
    require(sent2.size(Seq2seq.timeDim) == 1, "expect decoder input is batch x time(1) x feature")

    var curInput = sent2
    val sizes = curInput.size()
    val concat = Tensor[T](Array(sizes(0), maxSeqLen + 1) ++ sizes.drop(2))
    concat.narrow(Seq2seq.timeDim, 1, 1).copy(sent2)
    var break = false
    var j = 1
    // Iteratively output predicted words
    while (j <= maxSeqLen && !break) {
      val modelOutput = updateOutput(T(sent1, curInput)).toTensor[T]
      val generateOutput = if (infer != null) infer.forward(modelOutput) else modelOutput
      val predict = generateOutput.select(2, generateOutput.size(2))

      if (stopSign != null && predict.almostEqual(stopSign, 1e-8)) break = true
      j += 1
      concat.narrow(Seq2seq.timeDim, j, 1).copy(predict)
      curInput = concat.narrow(Seq2seq.timeDim, 1, j)
    }
    curInput
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
    if (preEncoder != null) pt.add(generator.getParametersTable())
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
    decoderOutput = null
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
     maskZero: Boolean = false,
     generator: AbstractModule[Activity, Activity, T] = null)
    (implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    new Seq2seq[T](encoderCells, decoderCells, preEncoder, preDecoder, bridges,
      maskZero, generator).build()
  }

  val timeDim = 2
}

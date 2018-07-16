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

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.BigDLWrapper
import com.intel.analytics.bigdl.optim.Predictor
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag
import com.intel.analytics.zoo.models.common.ZooModel
import org.apache.spark.rdd.RDD

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
  private var generator =
    Identity[T]().asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  private var inferenceModel: Boolean = false

  /**
   * @param maxLen max sequence length of output
   * @return this container
   */
  def setInferenceMode(maxLen: Int = 30, stopSign: Tensor[T] = null,
              generator: AbstractModule[Tensor[T], Tensor[T], T] = null): Unit = {
    seqLen = maxLen
    inferenceModel = true
    this.stopSign = stopSign
    this.generator = generator
  }

  def setTrain(): Unit = {
    inferenceModel = false
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

    output = if (!inferenceModel) {
      encoderOutput = encoder.forward(encoderInput).toTensor
      decoderInput = if (preDecoder != null) {
        val preDecoderOutput = preDecoder.forward(preDecoderInput).toTensor
        //      if (preDecoderOutput.dim() == preDecoderInput.dim + 1 && seqLen > 0) {
        //        preDecoderOutput.select(Seq2seq.timeDim, preDecoderOutput.size(Seq2seq.timeDim))
        //      } else preDecoderOutput
        preDecoderOutput
      } else preDecoderInput

      if (bridges != null) bridges.forwardStates(enc, dec)
      decoder.forward(decoderInput).toTensor
    } else {
      inference(input.toTable, seqLen, stopSign, generator)
    }

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

  /**
   * @param stopSign if prediction is the same with stopSign, it will stop predict
   */
  def inference(input: Table, maxSeqLen: Int = 30, stopSign: Tensor[T] = null,
              layer: AbstractModule[Tensor[T], Tensor[T], T] = null): Tensor[T] = {
    var break = false
    val sent1 = input.toTable[Tensor[T]](1)
    val sent2 = input.toTable[Tensor[T]](2)
    var predict: Tensor[T] = null
    // expect sent2 with time dim
    var pOutput = sent2

    encoder.forward(sent1).toTensor[T]
    if (bridges != null) bridges.forwardStates(enc, dec)
    var i = 1
    // Iteratively output predicted words
    while (i < maxSeqLen && !break) {
      val nextInput = if (preDecoder != null) {
        // expect pOutput with time dim
        preDecoder.forward(pOutput).toTensor
      } else pOutput
      // add time dim
//      val inputSize = nextInput.size()
//      nextInput.resize(Array(inputSize(0), 1) ++ inputSize.drop(1))

      // decoder0 without time dim
      val decoderO = decoder.forward(nextInput).toTensor[T].select(Seq2seq.timeDim, nextInput.size(2))
      val curOutput = if (layer != null)
        layer.forward(decoderO).toTensor[T]
      else decoderO

      if (stopSign != null && curOutput.almostEqual(stopSign, 1e-8)) break = true
      if (!break) {
        if (predict == null) {
          val sizes = curOutput.size()
          predict = Tensor[T](Array(sizes(0), maxSeqLen) ++ sizes.drop(1))
        }
        predict.narrow(Seq2seq.timeDim, i, 1).copy(curOutput)
      }
      pOutput = predict.narrow(Seq2seq.timeDim, i, 1)
      i += 1
    }
    if (predict != null) predict.narrow(Seq2seq.timeDim, 1, i) else sent2
  }

  private def insertTimeDim(tensor: Tensor[T], len: Int): Tensor[T] = {
    val sizes = tensor.size()
    tensor.resize(Array(sizes(0), len) ++ sizes.drop(1))
    tensor
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

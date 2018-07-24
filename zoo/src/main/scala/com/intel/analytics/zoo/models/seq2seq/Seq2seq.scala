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
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter.ArrayConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable, ModuleSerializer, SerializeContext}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag
import com.intel.analytics.zoo.models.common.ZooModel

import scala.collection.mutable.ArrayBuffer
import scala.reflect.runtime._

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
class Seq2seq[T: ClassTag](val encoderCells: Array[Cell[T]],
                           val decoderCells: Array[Cell[T]],
                           val preEncoder: AbstractModule[Activity, Activity, T] = null,
                           val preDecoder: AbstractModule[Activity, Activity, T] = null,
                           val bridges: Bridge = new PassThroughBridge(),
                           val maskZero: Boolean = false,
                           val generator: AbstractModule[Activity, Activity, T] = null)
  (implicit ev: TensorNumeric[T]) extends ZooModel[Activity, Tensor[T], T] {
  private var preDecoderInput: Tensor[T] = null
  private var decoderInput: Tensor[T] = null
  private var encoderInput: Tensor[T] = null
  private var encoderOutput: Tensor[T] = null
  private var decoderOutput: Tensor[T] = null
  private var enc: Array[ZooRecurrent[T]] = null
  private var dec: Array[ZooRecurrent[T]] = null

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

object Seq2seq extends ModuleSerializable {
  val timeDim = 2
  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.models.seq2seq.Seq2seq",
    Seq2seq)

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

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
//    println("hehe2")
//    val attrMap = context.bigdlModule.getAttrMap
//    val encoderCellsAttr = attrMap.get("encoderCells")
//    val encoderCells = DataConverter.
//      getAttributeValue(context, encoderCellsAttr).
//      asInstanceOf[Array[Cell[T]]]
//
//    val decoderCellsAttr = attrMap.get("decoderCells")
//    val decoderCells = DataConverter.
//      getAttributeValue(context, decoderCellsAttr).
//      asInstanceOf[Array[Cell[T]]]
//
//    val preEncoderAttr = attrMap.get("preEncoder")
//    val preEncoder = DataConverter.
//      getAttributeValue(context, preEncoderAttr).
//      asInstanceOf[AbstractModule[Activity, Activity, T]]
//
//    val preDecoderAttr = attrMap.get("preDecoder")
//    val preDecoder = DataConverter.
//      getAttributeValue(context, preDecoderAttr).
//      asInstanceOf[AbstractModule[Activity, Activity, T]]
//
//    val generatorAttr = attrMap.get("generator")
//    val generator = DataConverter.
//      getAttributeValue(context, generatorAttr).
//      asInstanceOf[AbstractModule[Activity, Activity, T]]
//
//    val maskZeroAttr = attrMap.get("maskZero")
//    val maskZero = DataConverter.
//      getAttributeValue(context, maskZeroAttr).
//      asInstanceOf[Boolean]
//
//    // Bridge deserailize
//    val bridgesAttr = attrMap.get("bridgesType")
//    val bridgeType = DataConverter.getAttributeValue(context, bridgesAttr)
//      .asInstanceOf[String]
//    val bridge = bridgeType match {
//      case "zero" => new ZeroBridge()
//      case "passthrough" => new PassThroughBridge()
//      case "initialstatebridge" => {
//        val activitationsAttr = attrMap.get("activations")
//        val activitationsFlat = DataConverter.getAttributeValue(context, activitationsAttr)
//          .asInstanceOf[Array[AbstractModule[_, _, T]]].map(_.asInstanceOf[TensorModule[T]])
//        val activitations = new ArrayBuffer[Array[TensorModule[T]]]()
//        var i = 0
//        while (i < activitationsFlat.size) {
//          activitations += activitationsFlat.slice(i, i + 2)
//          i += 2
//        }
//        new InitialStateBridge[T](activitations.toArray)
//      }
//    }

//    Seq2seq(encoderCells, decoderCells, preEncoder, preDecoder,
//      bridge, maskZero, generator).asInstanceOf[AbstractModule[Activity, Activity, T]]

    Seq2seq(Array(LSTM[Float](10, 10)).asInstanceOf[Array[Cell[Float]]],
      Array(LSTM[Float](10, 10)).asInstanceOf[Array[Cell[Float]]])
      .asInstanceOf[AbstractModule[Activity, Activity, T]]

  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              seq2seqBuilder : BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]) : Unit = {
//    println("hehe")
//    val seq2seq = context.moduleData.module.asInstanceOf[Seq2seq[T]]
//
//    val encoderRecsBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context,
//      encoderRecsBuilder, seq2seq.encoderCells,
//      universe.typeOf[Array[_ <:
//        AbstractModule[_ <: Activity, _ <:  Activity, _ <: Any]]]
//    )
//    seq2seqBuilder.putAttr("encoderCells", encoderRecsBuilder.build)
//
//    val decoderRecsBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context,
//      decoderRecsBuilder, seq2seq.decoderCells,
//      universe.typeOf[Array[_ <:
//        AbstractModule[_ <: Activity, _ <:  Activity, _ <: Any]]])
//    seq2seqBuilder.putAttr("decoderCells", decoderRecsBuilder.build)
//
//    val preEncoderBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context,
//      preEncoderBuilder, seq2seq.preEncoder,
//      ModuleSerializer.abstractModuleType)
//    seq2seqBuilder.putAttr("preEncoder", preEncoderBuilder.build)
//
//    val preDecoderBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context,
//      preDecoderBuilder, seq2seq.preDecoder,
//      ModuleSerializer.abstractModuleType)
//    seq2seqBuilder.putAttr("preDecoder", preDecoderBuilder.build)
//
//    val generatorBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context,
//      generatorBuilder, seq2seq.generator,
//      ModuleSerializer.abstractModuleType)
//    seq2seqBuilder.putAttr("generator", generatorBuilder.build)
//
//    val maskZeroBuilder = AttrValue.newBuilder
//    DataConverter.setAttributeValue(context,
//      maskZeroBuilder, seq2seq.maskZero,
//      scala.reflect.runtime.universe.typeOf[Boolean])
//    seq2seqBuilder.putAttr("maskZero", maskZeroBuilder.build)
//
//    // Bridge serailize
//    val bridgeTypeBuilder = AttrValue.newBuilder
//    if (seq2seq.bridges.isInstanceOf[ZeroBridge]) {
//      DataConverter.setAttributeValue(context,
//        bridgeTypeBuilder, "zero", scala.reflect.runtime.universe.typeOf[String])
//      seq2seqBuilder.putAttr("bridgesType", bridgeTypeBuilder.build)
//    } else if (seq2seq.bridges.isInstanceOf[PassThroughBridge]) {
//      DataConverter.setAttributeValue(context,
//        bridgeTypeBuilder, "passthrough", scala.reflect.runtime.universe.typeOf[String])
//      seq2seqBuilder.putAttr("bridgesType", bridgeTypeBuilder.build)
//    } else if (seq2seq.bridges.isInstanceOf[InitialStateBridge[T]]) {
//      DataConverter.setAttributeValue(context,
//        bridgeTypeBuilder, "initialstatebridge", scala.reflect.runtime.universe.typeOf[String])
//      seq2seqBuilder.putAttr("bridgesType", bridgeTypeBuilder.build)
//      val activationsBuilder = AttrValue.newBuilder
//      ArrayConverter.setAttributeValue(context, activationsBuilder,
//        seq2seq.bridges.asInstanceOf[InitialStateBridge[T]].activations.flatten,
//        scala.reflect.runtime.universe.typeOf[Array[_ <:
//          AbstractModule[_ <: Activity, _ <:  Activity, _ <: Any]]])
//      seq2seqBuilder.putAttr("activations", activationsBuilder.build)
//    }
  }
}

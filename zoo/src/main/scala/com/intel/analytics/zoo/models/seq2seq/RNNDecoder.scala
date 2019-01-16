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
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasLayerSerializable}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializer,
SerializeContext}
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter.ArrayConverter
import com.intel.analytics.zoo.pipeline.api.keras.layers.SelectTable
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime._

/**
 * [[RNNDecoder]] A generic recurrent neural network decoder
 *
 * @param rnns rnn layers used for decoder, support stacked rnn layers
 * @param embedding embedding layer in decoder
 * @param inputShape shape of input
 */
class RNNDecoder[T: ClassTag](val rnns: Array[Recurrent[T]],
  val embedding: KerasLayer[Tensor[T], Tensor[T], T],
  var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Decoder[T](inputShape) {

  private def checkStateShape(stateShape: Shape, hiddenSizes: Array[Int]): Boolean = {
    if (stateShape.isInstanceOf[SingleShape]) {
      if (hiddenSizes.length > 1) return false
      return stateShape.toSingle()(1) == hiddenSizes.head
    } else {
      var isMatch = true
      var i = 0
      while (i < stateShape.toMulti().size) {
        isMatch = isMatch && checkStateShape(stateShape.toMulti()(i), Array(hiddenSizes(i)))
        i += 1
      }
      isMatch
    }
  }

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Tensor[T], T] = {
    val layer = Sequential()

    // get decoder input
    layer.add(SelectTable(0, KerasUtils.removeBatch(inputShape)))
    if (embedding != null) layer.add(embedding)
    rnns.foreach(layer.add(_))

    val stateShape = inputShape.toMulti().last
    var i = 0
    while (i < rnns.size) {
      require(checkStateShape(stateShape.toMulti()(i), rnns(i).getHiddenShape()) == true,
        s"decoder init states shape should match decoder layers! " +
          s"Decoder layer expect hidden size (${rnns(i).getHiddenShape().mkString(" ")})," +
          s" which actually feed shape is ${stateShape.toMulti()(i)}. Please update decoder" +
          s" hidden size or update bridge/encoder hidden size")
      i += 1
    }
    layer.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    rnns.last.getOutputShape()
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    val states = input.toTable[Table](2)

    var i = 0
    while (i < rnns.size) {
      rnns(i).setHiddenState(states(i + 1))
      i += 1
    }

    output = labor.updateOutput(input)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    labor.updateGradInput(input, gradOutput)
    val gradStates = rnns.map(_.getGradHiddenState())
    val rnnsGradInput = Tensor[T](input.toTable[Tensor[T]](1).size())

    gradInput = T(rnnsGradInput, T.array(gradStates))
    gradInput
  }
}

object RNNDecoder extends KerasLayerSerializable {
  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.models.seq2seq.RNNDecoder",
    RNNDecoder)

  /**
   * [[RNNDecoder]] A generic recurrent neural network decoder
   *
   * @param rnns rnn layers used for decoder, support stacked rnn layers
   * @param embedding embedding layer in decoder
   * @param inputShape shape of input
   */
  def apply[@specialized(Float, Double) T: ClassTag](rnns: Array[Recurrent[T]],
    embedding: KerasLayer[Tensor[T], Tensor[T], T],
    inputShape: Shape)(implicit ev: TensorNumeric[T]): RNNDecoder[T] = {
    new RNNDecoder[T](rnns, embedding, inputShape)
  }

  /**
   * [[RNNDecoder]] A generic recurrent neural network decoder
   *
   * @param rnnType style of recurrent unit, one of [SimpleRNN, LSTM, GRU]
   * @param numLayers number of layers used in decoder
   * @param hiddenSize hidden size of decoder
   * @param embedding embedding layer in decoder
   * @param inputShape shape of input
   */
  def apply[@specialized(Float, Double) T: ClassTag](rnnType: String,
    numLayers: Int,
    hiddenSize: Int,
    embedding: KerasLayer[Tensor[T], Tensor[T], T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): RNNDecoder[T] = {
    val rnn = new ArrayBuffer[Recurrent[T]]()
    rnnType.toLowerCase() match {
      case "lstm" =>
        for (i <- 1 to numLayers) rnn.append(LSTM(hiddenSize, returnSequences = true))
      case "gru" =>
        for (i <- 1 to numLayers) rnn.append(GRU(hiddenSize, returnSequences = true))
      case "simplernn" =>
        for (i <- 1 to numLayers) rnn.append(SimpleRNN(hiddenSize, returnSequences = true))
      case _ => throw new IllegalArgumentException(s"Please use " +
        s"RNNDecoder(rnn: Array[Recurrent[T]], embedding: KerasLayer[Tensor[T], Tensor[T], T])" +
        s"to create a decoder")
    }
    RNNDecoder[T](rnn.toArray, embedding, inputShape)
  }

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap

    val rnns = DataConverter.getAttributeValue(context, attrMap.get("rnns")).
      asInstanceOf[Array[AbstractModule[_, _, T]]].map(_.asInstanceOf[Recurrent[T]])
    rnns.map(_.labor = null)
    rnns.map(_.modules.clear())

    val embeddingAttr = attrMap.get("embedding")
    val embedding = DataConverter.getAttributeValue(context, embeddingAttr).
      asInstanceOf[KerasLayer[Tensor[T], Tensor[T], T]]
    if (embedding != null) {
      embedding.labor = null
      embedding.modules.clear()
    }

    val shapeAttr = attrMap.get("shape")
    val shape = DataConverter.getAttributeValue(context, shapeAttr).asInstanceOf[Shape]
    val decoder = RNNDecoder(rnns, embedding, shape)

    decoder.build(KerasUtils.addBatch(shape))

    decoder.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    decoderBuilder : BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]) : Unit = {

    val decoder = context.moduleData.module.asInstanceOf[RNNDecoder[T]]

    val rnnsBuilder = AttrValue.newBuilder
    ArrayConverter.setAttributeValue(context, rnnsBuilder,
      context.moduleData.module.asInstanceOf[RNNDecoder[T]].rnns,
      scala.reflect.runtime.universe.typeOf[Array[_ <:
        AbstractModule[_ <: Activity, _ <:  Activity, _ <: Any]]])
    decoderBuilder.putAttr("rnns", rnnsBuilder.build)

    val embeddingBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, embeddingBuilder,
      decoder.embedding, ModuleSerializer.abstractModuleType)
    decoderBuilder.putAttr("embedding", embeddingBuilder.build)

    val shapeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, shapeBuilder,
      decoder.inputShape, universe.typeOf[Shape])
    decoderBuilder.putAttr("shape", shapeBuilder.build)

    appendKerasLabel(context, decoderBuilder)
  }
}

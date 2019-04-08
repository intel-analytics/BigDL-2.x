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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.{Module, RandomNormal, StaticGraph}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasLayerSerializable}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleData, ModuleSerializer, SerializeContext}
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, Table}
import com.intel.analytics.zoo.models.seq2seq.RNNEncoder._
import com.intel.analytics.zoo.models.seq2seq.{RNNDecoder, RNNEncoder}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd.{Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.{GraphRef, KerasUtils}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.models.Model.{apply => _, _}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.reflect.runtime._

/**
 * [[BERT]] A self attention keras like layer.
 * Input is a Table which consists of 4 tensors.
 * 1. Token id tensor: shape [batch, seqLen] with the word token indices in the vocabulary
 * 2. Token type id tensor: shape [batch, seqLen] with the token types in [0, 1].
 *    0 menas `sentence A` and 1 means a `sentence B` (see BERT paper for more details).
 * 3. Position id tensor: shape [batch, seqLen] with positions in the sentence.
 * 4. Attention_mask tensor: shape [batch, seqLen] with indices in [0, 1].
 *   It's a mask to be used if the input sequence length is smaller than seqLen in
 *   the current batch.
 * Output is an Activity which output the states of BERT layer
 * @param nBlock block number
 * @param nHead head number
 * @param intermediateSize The size of the "intermediate" (i.e., feed-forward)
 * @param hiddenPDrop The dropout probabilitiy for all fully connected layers
 * @param attnPDrop drop probability of attention
 * @param initializerRange weight initialization range
 * @param outputAllBlock whether output all blocks' output
 * @param embeddingLayer embedding layer
 * @param inputShape input shape, default is null
 */
class BERT[T: ClassTag] (
  nBlock: Int = 12,
  nHead: Int = 12,
  intermediateSize: Int = 3072,
  hiddenPDrop: Double = 0.1,
  attnPDrop: Double = 0.1,
  initializerRange: Double = 0.02,
  outputAllBlock: Boolean = true,
  embeddingLayer: KerasLayer[Activity, Tensor[T], T],
  inputShape: Shape)(implicit ev: TensorNumeric[T])
  extends TransformerLayer[T](nBlock, hiddenPDrop, attnPDrop, nHead,
    initializerRange, true, outputAllBlock, embeddingLayer, intermediateSize, inputShape)
  with Net {

  override def projectionLayer(outputSize: Int): Net = {
    new Dense(outputSize, init = RandomNormal(0.0, initializerRange))
  }

  override def buildInput(inputShape: Shape):
  (Variable[T], List[Variable[T]], List[Variable[T]]) = {
    require(inputShape.isInstanceOf[MultiShape] &&
      inputShape.asInstanceOf[MultiShape].value.size == 4, "BERT input must be" +
      " a list of 4 tensors (consisting of input sequence, sequence positions," +
      "segment id, attention mask)")
    val _inputShape = KerasUtils.removeBatch(inputShape).toMulti()
    seqLen = _inputShape.head.toSingle().head

    val inputs = _inputShape.map(Variable(_))
    return ((- inputs.last + 1.0) * -10000.0, inputs.dropRight(1), inputs)
  }
}

object BERT extends KerasLayerSerializable {
  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.pipeline.api.keras.layers.BERT",
    BERT)

  /**
   * [[BERT]] A self attention keras like layer
   * @param vocab vocabulary size of training data, default is 40990
   * @param hiddenSize ize of the encoder layers, default is 768
   * @param nBlock block number, default is 12
   * @param nHead head number, default is 12
   * @param seqLen sequence length, default is 512
   * @param intermediateSize The size of the "intermediate" (i.e., feed-forward), default is 3072
   * @param hiddenPDrop The dropout probabilitiy for all fully connected layers, default is 0.1
   * @param attnPDrop drop probability of attention, default is 0.1
   * @param initializerRange weight initialization range, default is 0.02
   * @param outputAllBlock whether output all blocks' output, default is true
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    vocab: Int = 40990,
    hiddenSize: Int = 768,
    nBlock: Int = 12,
    nHead: Int = 12,
    seqLen: Int = 512,
    intermediateSize: Int = 3072,
    hiddenPDrop: Double = 0.1,
    attnPDrop: Double = 0.1,
    initializerRange: Double = 0.02,
    outputAllBlock: Boolean = true
    )(implicit ev: TensorNumeric[T]): BERT[T] = {
    require(hiddenSize > 0, "hiddenSize must be great" +
      "than 0 with default embedding layer")

    val wordInput = Variable(Shape(seqLen))
    val tokenTypeInput = Variable(Shape(seqLen))
    val positionInput = Variable(Shape(seqLen))

    val wordEmbeddings = new Embedding(vocab, hiddenSize, inputShape = Shape(seqLen),
      init = RandomNormal(0.0, initializerRange)).from(wordInput)
    val positionEmbeddings = new Embedding(seqLen, hiddenSize, inputShape = Shape(seqLen),
      init = RandomNormal(0.0, initializerRange)).from(positionInput)
    val tokenTypeEmbeddings = new Embedding(2, hiddenSize, inputShape = Shape(seqLen),
      init = RandomNormal(0.0, initializerRange)).from(tokenTypeInput)

    val embeddings = wordEmbeddings + positionEmbeddings + tokenTypeEmbeddings
    val w = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor.ones[T](hiddenSize).view(1, hiddenSize))
    val b = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor[T](hiddenSize).view(1, hiddenSize))
    val afterNorm = TransformerLayer.layerNorm(embeddings, 1e-12, weight = w, bias = b)
    val h = Dropout(hiddenPDrop).from(afterNorm)

    val embeddingLayer = Model(Array(wordInput, tokenTypeInput, positionInput), h)
    new BERT[T](nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop, initializerRange,
    outputAllBlock, embeddingLayer.asInstanceOf[KerasLayer[Activity, Tensor[T], T]],
      Shape(List(Shape(seqLen), Shape(seqLen), Shape(seqLen), Shape(1, 1, seqLen))))
  }

  /**
   * [[BERT]] A self attention keras like layer
   * @param nBlock block number
   * @param nHead head number
   * @param intermediateSize The size of the "intermediate" (i.e., feed-forward)
   * @param hiddenPDrop The dropout probabilitiy for all fully connected layers
   * @param attnPDrop drop probability of attention
   * @param initializerRange weight initialization range
   * @param outputAllBlock whether output all blocks' output
   * @param embeddingLayer embedding layer
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    nBlock: Int,
    nHead: Int,
    intermediateSize: Int,
    hiddenPDrop: Double,
    attnPDrop: Double,
    initializerRange: Double,
    outputAllBlock: Boolean,
    embeddingLayer: KerasLayer[Activity, Tensor[T], T])
    (implicit ev: TensorNumeric[T]): BERT[T] = {
    new BERT[T](nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop, initializerRange,
      outputAllBlock, embeddingLayer, null)
  }

  /**
   * Load an existing model (with weights).
   *
   * @param path The path for the pre-defined model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def loadModel[T: ClassTag](path: String,
    weightPath: String = null)(implicit ev: TensorNumeric[T]): BERT[T] = {
    Module.loadModule[T](path, weightPath).asInstanceOf[BERT[T]]
  }

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap

    val nBlockAttr = attrMap.get("nBlock")
    val nBlock =
      DataConverter.getAttributeValue(context, nBlockAttr)
        .asInstanceOf[Int]

    val nHeadAttr = attrMap.get("nHead")
    val nHead =
      DataConverter.getAttributeValue(context, nHeadAttr)
        .asInstanceOf[Int]

    val intermediateSizeAttr = attrMap.get("intermediateSize")
    val intermediateSize =
      DataConverter.getAttributeValue(context, intermediateSizeAttr)
        .asInstanceOf[Int]

    val hiddenPDropAttr = attrMap.get("hiddenPDrop")
    val hiddenPDrop =
      DataConverter.getAttributeValue(context, hiddenPDropAttr)
        .asInstanceOf[Double]

    val attnPDropAttr = attrMap.get("attnPDrop")
    val attnPDrop =
      DataConverter.getAttributeValue(context, attnPDropAttr)
        .asInstanceOf[Double]

    val initializerRangeAttr = attrMap.get("initializerRange")
    val initializerRange =
      DataConverter.getAttributeValue(context, initializerRangeAttr)
        .asInstanceOf[Double]

    val outputAllBlockAttr = attrMap.get("outputAllBlock")
    val outputAllBlock =
      DataConverter.getAttributeValue(context, outputAllBlockAttr)
        .asInstanceOf[Boolean]

    import scala.collection.JavaConverters._
    val subProtoModules = context.bigdlModule.getSubModulesList.asScala
    val subModules = subProtoModules.map(module => {
      val subModuleData = ModuleSerializer.load(DeserializeContext(module,
        context.storages, context.storageType, _copyWeightAndBias))
      subModuleData.module
    })
    val tGraph = subModules(0).asInstanceOf[StaticGraph[T]]
    val embeddingLayer = Model(tGraph.inputs.toArray, new GraphRef(tGraph).getOutputs().toArray)

    val shapeAttr = attrMap.get("seqLen")
    val seqLen = DataConverter.getAttributeValue(context, shapeAttr).asInstanceOf[Int]

    val shape = Shape(List(Shape(seqLen), Shape(seqLen), Shape(seqLen), Shape(1, 1, seqLen)))
    val bert = BERT(nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop,
      initializerRange, outputAllBlock,
      embeddingLayer.asInstanceOf[KerasLayer[Activity, Tensor[T], T]])

    bert.build(KerasUtils.addBatch(shape))
    bert.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    bertBuilder : BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]) : Unit = {

    val bert = context.moduleData.module.asInstanceOf[BERT[T]]

    val nBlockBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, nBlockBuilder,
      bert.nBlock, universe.typeOf[Int])
    bertBuilder.putAttr("nBlock", nBlockBuilder.build)

    val nHeadBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, nHeadBuilder,
      bert.nHead, universe.typeOf[Int])
    bertBuilder.putAttr("nHead", nHeadBuilder.build)

    val intermediateSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, intermediateSizeBuilder,
      bert.intermediateSize, universe.typeOf[Int])
    bertBuilder.putAttr("intermediateSize", intermediateSizeBuilder.build)

    val hiddenPDropBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, hiddenPDropBuilder,
      bert.hiddenPDrop, universe.typeOf[Double])
    bertBuilder.putAttr("hiddenPDrop", hiddenPDropBuilder.build)

    val attnPDropBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, attnPDropBuilder,
      bert.attnPDrop, universe.typeOf[Double])
    bertBuilder.putAttr("attnPDrop", attnPDropBuilder.build)

    val initializerRangeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, initializerRangeBuilder,
      bert.initializerRange, universe.typeOf[Double])
    bertBuilder.putAttr("initializerRange", initializerRangeBuilder.build)

    val outputAllBlockBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, outputAllBlockBuilder,
      bert.outputAllBlock, universe.typeOf[Boolean])
    bertBuilder.putAttr("outputAllBlock", outputAllBlockBuilder.build)

    val embLabor = bert.embeddingLayer.labor.asInstanceOf[AbstractModule[Activity, Activity, T]]
    val subModule = ModuleSerializer.serialize(SerializeContext(ModuleData(embLabor,
      new ArrayBuffer[String](), new ArrayBuffer[String]()), context.storages,
      context.storageType, _copyWeightAndBias))
    bertBuilder.addSubModules(subModule.bigDLModule)

    val seqLenBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, seqLenBuilder,
      bert.seqLen, universe.typeOf[Int])
    bertBuilder.putAttr("seqLen", seqLenBuilder.build)

    appendKerasLabel(context, bertBuilder)
  }
}

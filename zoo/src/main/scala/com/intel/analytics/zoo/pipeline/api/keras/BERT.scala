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

package com.intel.analytics.zoo.pipeline.api.keras

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, Table}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

import scala.reflect.ClassTag

class BERT[T: ClassTag] (
  val hiddenSize: Int = 768,
  val nBlock: Int = 12,
  val nHead: Int = 12,
  val intermediateSize: Int = 3072,
  val hiddenPDrop: Double = 0.1,
  val attnPDrop: Double = 0.1,
  val outputAllBlock: Boolean = true,
  val embeddingLayer: KerasLayer[Activity, Tensor[T], T],
  var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Table, Tensor[T], T](KerasUtils.addBatch(inputShape))
  with Net {
  require(hiddenSize >= nHead, "nHead cannot be larger than hiddenSize!")

  override def doBuild(inputShape: Shape): AbstractModule[Table, Tensor[T], T] = {
    require(inputShape.isInstanceOf[MultiShape], "TransformerLayer input must" +
      " be a multiple shape")
    val _inputShape = KerasUtils.removeBatch(inputShape)
    val seqLen = _inputShape.toMulti().head.toSingle().head

    val wordInput = Variable(Shape(seqLen))
    val tokenTypeInput = Variable(Shape(seqLen))
    val positionInput = Variable(Shape(seqLen))

    val attentionMask = Variable(Shape(1, 1, seqLen))

    require(embeddingLayer.isInstanceOf[Net], "use layers from" +
      "com.intel.analytics.zoo.pipeline.api.keras and operators from" +
      " com.intel.analytics.zoo.pipeline.api.autograd to construct the embedding layer")
    val embedding = embeddingLayer.asInstanceOf[Net]
    val e = embedding.from(wordInput, tokenTypeInput, positionInput)

    val nextInput: Variable[T] = e

    // need user input attention_mask, and shape into the right dim
    val extended_attention_mask = (- attentionMask + 1.0) * -10000.0

    val modelOutputSize = nBlock
    val modelOutput = new Array[Variable[T]](modelOutputSize)
    modelOutput(0) = block(nextInput, extended_attention_mask)

    for (i <- 1 until nBlock) {
      val output = block(modelOutput(i - 1), extended_attention_mask)
      modelOutput(i) = output
    }

    val model = if (outputAllBlock) {
        Model(Array(wordInput, tokenTypeInput, positionInput, attentionMask), modelOutput)
    } else Model(Array(wordInput, tokenTypeInput, positionInput, attentionMask), modelOutput.last)

    model.asInstanceOf[AbstractModule[Table, Tensor[T], T]]
  }

  def block(x: Variable[T], attention_mask: Variable[T] = null): Variable[T] = {
    // g, b for layerNorm
    val g = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor.ones[T](hiddenSize).view(1, hiddenSize))
    val b = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor[T](hiddenSize).view(1, hiddenSize))

    // g, b for layerNorm
    val g2 = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor.ones[T](hiddenSize).view(1, hiddenSize))
    val b2 = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor[T](hiddenSize).view(1, hiddenSize))

    val a = attention(x, attention_mask)
    // BertOutput2
    val n = TransformerLayer.layerNorm(x + a, e=1e-12, weight=g2, bias=b2)

    val m = mlp(n)
    // Bertoutput2
    val h = TransformerLayer.layerNorm(n + m, e=1e-12, weight=g, bias=b)
    h
  }

  def mlp(x: Variable[T]): Variable[T] = {
    // BertIntermediate
    val h = Dense(intermediateSize).from(x)
    val a = TransformerLayer.gelu(h)

    // Bertoutput1
    val h2 = Dense(hiddenSize).from(a)
    Dropout(hiddenPDrop).from(h2)
  }

  def attention(x: Variable[T], attention_mask: Variable[T]): Variable[T] = {
    val attention_head_size = hiddenSize / nHead
    val all_head_size = nHead * attention_head_size
    val query = Dense(all_head_size).from(x)
    val key = Dense(all_head_size).from(x)
    val value = Dense(all_head_size).from(x)
    val q = TransformerLayer.splitHeads(query, nHead)
    val k = TransformerLayer.splitHeads(key, nHead, k = true)
    val v = TransformerLayer.splitHeads(value, nHead)
    val a = attn(q, k, v, attention_mask, true) // m: (-1, 12, 77, 64)
    val m = TransformerLayer.mergeHeads(a) // m: (-1, 77, 768)
    // Bertoutput1
    val n = Dense(hiddenSize).from(m) // n: (-1, 77, 768)
    Dropout(attnPDrop).from(n)
  }

  // scale shoule be set in Attention
  def attn(q: Variable[T], k: Variable[T], v: Variable[T], attention_mask: Variable[T],
    scale: Boolean = false): Variable[T] = {
    // q:(16, 12, 77, 64) k:(16, 12, 64, 77) v:(16, 12, 77, 64)
    var w = AutoGrad.mm(q, k) // w: (16, 12, 77, 77)
    if (scale) w = w / scala.math.sqrt(v.getOutputShape().toSingle().toArray.last)

    if (attention_mask != null) {
      w = w + attention_mask
    }
    w = Activation[Float]("softmax").from(w)
    w = AutoGrad.mm(w, v)

    w = Dropout(hiddenPDrop).from(w)
    w
  }
}

object BERT {
  def apply[@specialized(Float, Double) T: ClassTag](
    vocab: Int = 40990,
    hiddenSize: Int = 768,
    nBlock: Int = 12,
    nHead: Int = 12,
    seqLen: Int = 512,
    intermediateSize: Int = 3072,
    hiddenPDrop: Double = 0.1,
    attnPDrop: Double = 0.1,
    outputAllBlock: Boolean = true
    )(implicit ev: TensorNumeric[T]): BERT[T] = {
    require(hiddenSize > 0, "hiddenSize must be great" +
      "than 0 with default embedding layer")

    val wordInput = Variable(Shape(seqLen))
    val tokenTypeInput = Variable(Shape(seqLen))
    val positionInput = Variable(Shape(seqLen))

    val wordEmbeddings = Embedding(vocab, hiddenSize, inputLength = seqLen).from(wordInput)
    val positionEmbeddings = Embedding(seqLen, hiddenSize, inputLength = seqLen)
      .from(positionInput)
    val tokenTypeEmbeddings = Embedding(2, hiddenSize, inputLength = seqLen)
      .from(tokenTypeInput)

    val embeddings = wordEmbeddings + positionEmbeddings + tokenTypeEmbeddings
    // g, b for layerNorm
    val embeddingG = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor.ones[T](hiddenSize).view(1, hiddenSize))
    val embeddingB = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor[T](hiddenSize).view(1, hiddenSize))
    val afterNorm = TransformerLayer.layerNorm(embeddings, weight=embeddingG, bias=embeddingB)
    val h = Dropout(hiddenPDrop).from(afterNorm)

    val embeddingLayer = Model(Array(wordInput, tokenTypeInput, positionInput), h)
    new BERT[T](hiddenSize, nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop,
    outputAllBlock, embeddingLayer.asInstanceOf[KerasLayer[Activity, Tensor[T], T]])
  }

  def apply[@specialized(Float, Double) T: ClassTag](
    hiddenSize: Int,
    nBlock: Int,
    nHead: Int,
    intermediateSize: Int,
    hiddenPDrop: Double,
    attnPDrop: Double,
    outputAllBlock: Boolean,
    embeddingLayer: KerasLayer[Activity, Tensor[T], T])
    (implicit ev: TensorNumeric[T]): BERT[T] = {
    new BERT[T](hiddenSize, nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop,
      outputAllBlock, embeddingLayer)
  }
}

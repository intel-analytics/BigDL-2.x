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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd.{Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

import scala.reflect.ClassTag

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
  outputAllBlock: Boolean = true,
  embeddingLayer: KerasLayer[Activity, Tensor[T], T],
  inputShape: Shape)(implicit ev: TensorNumeric[T])
  extends TransformerLayer[T](nBlock, hiddenPDrop, attnPDrop, nHead, intermediateSize,
    true, outputAllBlock, embeddingLayer, inputShape)
  with Net {

  override def projectLayers(outputSize: Int, config: Double*): Net = {
    Dense(outputSize)
  }
}

object BERT {
  /**
   * [[BERT]] A self attention keras like layer
   * @param vocab vocabulary size of training data, default is 40990
   * @param hiddenSize ize of the encoder layers, default is 768
   * @param nBlock block number
   * @param nHead head number
   * @param seqLen sequence lenght
   * @param intermediateSize The size of the "intermediate" (i.e., feed-forward)
   * @param hiddenPDrop The dropout probabilitiy for all fully connected layers
   * @param attnPDrop drop probability of attention
   * @param outputAllBlock whether output all blocks' output
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
    val w = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor.ones[T](hiddenSize).view(1, hiddenSize))
    val b = Parameter[T](Shape(1, hiddenSize),
      initWeight = Tensor[T](hiddenSize).view(1, hiddenSize))
    val afterNorm = TransformerLayer.layerNorm(embeddings, 1e-12, weight = w, bias = b)
    val h = Dropout(hiddenPDrop).from(afterNorm)

    val embeddingLayer = Model(Array(wordInput, tokenTypeInput, positionInput), h)
    new BERT[T](nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop,
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
   * @param outputAllBlock whether output all blocks' output
   * @param embeddingLayer embedding layer
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    nBlock: Int,
    nHead: Int,
    intermediateSize: Int,
    hiddenPDrop: Double,
    attnPDrop: Double,
    outputAllBlock: Boolean,
    embeddingLayer: KerasLayer[Activity, Tensor[T], T])
    (implicit ev: TensorNumeric[T]): BERT[T] = {
    new BERT[T](nBlock, nHead, intermediateSize, hiddenPDrop, attnPDrop,
      outputAllBlock, embeddingLayer, null)
  }
}

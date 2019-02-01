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

package com.intel.analytics.zoo.models.bert

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag

class BERT[T: ClassTag] private(
   vocab: Int,
   hidden_size: Int = 768,
   num_hidden_layers: Int = 12,
   num_attention_heads: Int = 12,
   intermediate_size: Int = 3072,
   hidden_dropout_prob: Double = 0.1,
   attention_probs_dropout_prob: Double = 0.1,
   maxPositionEmbeddings: Int = 512,
//   typeVocabSize: Int = 2,
   output_all_encoded_layers: Boolean = true)(implicit ev: TensorNumeric[T])
  extends ZooModel[Activity, Activity, T] {
  require(hidden_size >= num_attention_heads, "num_attention_heads cannot be" +
    "larger than hidden_size!")

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    // (sequence length)
    val wordInput = Variable(Shape(maxPositionEmbeddings))
    val tokenTypeInput = Variable(Shape(maxPositionEmbeddings))
    val positionInput = Variable(Shape(maxPositionEmbeddings))
    val attentionMask = Variable(Shape(1, 1, maxPositionEmbeddings))

    val wordEmbeddings = Embedding(vocab, hidden_size, inputLength = maxPositionEmbeddings).from(wordInput)
    val positionEmbeddings = Embedding(maxPositionEmbeddings, hidden_size, inputLength = maxPositionEmbeddings)
      .from(positionInput)
    val tokenTypeEmbeddings = Embedding(2, hidden_size, inputLength = maxPositionEmbeddings)
      .from(tokenTypeInput)

    val embeddings = wordEmbeddings + positionEmbeddings + tokenTypeEmbeddings
    val afterNorm = layerNorm(embeddings, weight=embeddingG, bias=embeddingB)
    val h = Dropout(hidden_dropout_prob).from(afterNorm)

    val nextInput: Variable[T] = h

    // need user input attention_mask, and shape into the right dim
    val extended_attention_mask = (- attentionMask + 1.0) * -10000.0

    val modelOutputSize = num_hidden_layers
    val modelOutput = new Array[Variable[T]](modelOutputSize)
    modelOutput(0) = block(nextInput, extended_attention_mask)

    // TODO: CLONE for each num_hidden_layers
    for (i <- 1 until num_hidden_layers) {
      val output = block(modelOutput(i - 1), extended_attention_mask)
      modelOutput(i) = output
    }

    val model = if (output_all_encoded_layers) {
      Model(Array(wordInput, tokenTypeInput, positionInput, attentionMask), modelOutput)
    } else Model(Array(wordInput, tokenTypeInput, positionInput, attentionMask), modelOutput.last)

    model
  }

  def block(x: Variable[T], attention_mask: Variable[T] = null): Variable[T] = {
    val a = attention(x, attention_mask)
    // BertOutput2
    val n = layerNorm(x + a, e=1e-12, weight=g2, bias=b2)

    val m = mlp(n)
    // Bertoutput2
    val h = layerNorm(n + m, e=1e-12, weight=g, bias=b)
    h
  }

  def mlp(x: Variable[T]): Variable[T] = {
    // BertIntermediate
    val h = Dense(intermediate_size).from(x)
    val a = gelu(h)

    // Bertoutput1
    val h2 = Dense(hidden_size).from(a)
    Dropout(hidden_dropout_prob).from(h2)
  }


  // g, b for layerNorm
  val embeddingG = Parameter[T](Shape(1, hidden_size),
    initWeight = Tensor.ones[T](hidden_size).view(1, hidden_size))
  val embeddingB = Parameter[T](Shape(1, hidden_size),
    initWeight = Tensor[T](hidden_size).view(1, hidden_size))

  def layerNorm(x: Variable[T], e: Double = 1e-5, weight: Parameter[T], bias: Parameter[T]): Variable[T] = {
    val sizes = x.getOutputShape().toSingle().toArray
    val u = AutoGrad.mean(x, sizes.size - 1, true)
    val s = AutoGrad.mean(AutoGrad.square(x - u), sizes.size - 1, true)
    val y = (x - u) / AutoGrad.sqrt(s + e)
    y * weight + bias
  }

  // g, b for layerNorm
  val g = Parameter[T](Shape(1, hidden_size),
    initWeight = Tensor.ones[T](hidden_size).view(1, hidden_size))
  val b = Parameter[T](Shape(1, hidden_size),
    initWeight = Tensor[T](hidden_size).view(1, hidden_size))

  // g, b for layerNorm
  val g2 = Parameter[T](Shape(1, hidden_size),
    initWeight = Tensor.ones[T](hidden_size).view(1, hidden_size))
  val b2 = Parameter[T](Shape(1, hidden_size),
    initWeight = Tensor[T](hidden_size).view(1, hidden_size))

  def gelu(x: Variable[T]): Variable[T] = {
    x * 0.5 * (Activation("tanh").from((AutoGrad.square(x) * x * 0.044715 + x)
      * (scala.math.sqrt(2 / scala.math.Pi))) + 1)
  }

  def attention(x: Variable[T], attention_mask: Variable[T]): Variable[T] = {
    val attention_head_size = hidden_size / num_attention_heads
    val all_head_size = num_attention_heads * attention_head_size
    val query = Dense(all_head_size).from(x)
    val key = Dense(all_head_size).from(x)
    val value = Dense(all_head_size).from(x)
    val q = splitHeads(query)
    val k = splitHeads(key, k = true)
    val v = splitHeads(value)
    val a = attn(q, k, v, attention_mask, true) // m: (-1, 12, 77, 64)
    val m = mergeHeads(a) // m: (-1, 77, 768)
    // Bertoutput1
    val n = Dense(hidden_size).from(m) // n: (-1, 77, 768)
    Dropout(attention_probs_dropout_prob).from(n)
  }

  def splitHeads(x: Variable[T], k: Boolean = false): Variable[T] = {
    val sizes = x.getOutputShape().toSingle().toArray
    val newSizes = sizes.drop(1).dropRight(1) ++ Array(num_attention_heads, sizes.last / num_attention_heads)
    val r = Reshape(newSizes).from(x)
    if (k) Permute(Array(2, 3, 1)).from(r)
    else Permute(Array(2, 1, 3)).from(r)
  }

  def mergeHeads(x: Variable[T]): Variable[T] = {
    val p = AutoGrad.contiguous(Permute[T](Array(2, 1, 3)).from(x))
    val sizes = p.getOutputShape().toSingle().toArray
    Reshape(sizes.drop(1).dropRight(2) ++ Array(sizes.last * sizes(sizes.length - 2))).from(p)
  }

  // scale shoule be set in Attention
  def attn(q: Variable[T], k: Variable[T], v: Variable[T], attention_mask: Variable[T], scale: Boolean = false): Variable[T] = {
    // q:(16, 12, 77, 64) k:(16, 12, 64, 77) v:(16, 12, 77, 64)
    var w = AutoGrad.mm(q, k) // w: (16, 12, 77, 77)
    if (scale) w = w / scala.math.sqrt(v.getOutputShape().toSingle().toArray.last)

    if (attention_mask != null) {
      w = w + attention_mask
    }
    w = Activation[Float]("softmax").from(w)
    w = AutoGrad.mm(w, v)

    w = Dropout(hidden_dropout_prob).from(w)
    w
  }
}

object BERT {
  def apply[@specialized(Float, Double) T: ClassTag](
    vocab: Int,
    hidden_size: Int = 768,
    num_hidden_layers: Int = 12,
    num_attention_heads: Int = 12,
    intermediate_size: Int = 3072,
    hidden_dropout_prob: Double = 0.1,
    attention_probs_dropout_prob: Double = 0.1,
    maxPositionEmbeddings: Int = 512,
//    typeVocabSize: Int = 2,
    output_all_encoded_layers: Boolean = true)(implicit ev: TensorNumeric[T]): BERT[T] = {
    new BERT[T](vocab, hidden_size, num_hidden_layers, num_attention_heads,
      intermediate_size, hidden_dropout_prob, attention_probs_dropout_prob, maxPositionEmbeddings,
      output_all_encoded_layers).build()
  }

  def loadModel[T: ClassTag](
                              path: String,
                              weightPath: String = null)(implicit ev: TensorNumeric[T]): BERT[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[BERT[T]]
  }
}

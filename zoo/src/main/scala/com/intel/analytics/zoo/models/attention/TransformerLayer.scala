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

package com.intel.analytics.zoo.models.attention

import com.intel.analytics.bigdl.nn.RandomNormal
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, SingleShape}
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

class TransformerLayer[T: ClassTag] (
  val vocab: Int,
  val nCtx: Int,
  val embeddingSize: Int,
  val embeddingDrop: Double,
  val nLayer: Int,
//  afn: String,
  val residPdrop: Double,
  val attnPdrop: Double,
  val nHead: Int,
  var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))
    with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    // (sequence length, position)
    require(inputShape.isInstanceOf[SingleShape], "TransformerLayer input must" +
      " be a single shape")
    val _inputShape = KerasUtils.removeBatch(inputShape)
    require(_inputShape.equals(Shape(nCtx, 2)), "TransformerLayer input shape" +
      " must be Shape(nCtx, 2)")
    val input = Variable(_inputShape)
    val r = Reshape(Array(nCtx * 2)).from(input)
    val e = Embedding(vocab, embeddingSize, inputLength = nCtx * 2).from(r)

    val r2 = Reshape(Array(nCtx, 2, embeddingSize)).from(e)

    val h = AutoGrad.sum(r2, 2, false)

    var nextInput: Variable[T] = h

    for (i <- 0 until nLayer) {
      val output = block(nextInput)
      nextInput = output
    }

    val model = Model(input, nextInput)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }

  def block(x: Variable[T]): Variable[T] = {
    // g, b for layerNorm
    val g = Parameter[T](Shape(1, embeddingSize),
      initWeight = Tensor.ones[T](embeddingSize).view(1, embeddingSize))
    val b = Parameter[T](Shape(1, embeddingSize),
      initWeight = Tensor[T](embeddingSize).view(1, embeddingSize))

    // g, b for layerNorm
    val g2 = Parameter[T](Shape(1, embeddingSize),
      initWeight = Tensor.ones[T](embeddingSize).view(1, embeddingSize))
    val b2 = Parameter[T](Shape(1, embeddingSize),
      initWeight = Tensor[T](embeddingSize).view(1, embeddingSize))
    val a = attention(x)
    val n = layerNorm(x + a, weight = g, bias = b)
    val m = mlp(n)
    val h = layerNorm(n + m, weight = g2, bias = b2)
    h
  }


  def layerNorm(x: Variable[T], e: Double = 1e-5, weight: Parameter[T],
                bias: Parameter[T]): Variable[T] = {
    val sizes = x.getOutputShape().toSingle().toArray
    val u = AutoGrad.mean(x, sizes.size - 1, true)
    val s = AutoGrad.mean(AutoGrad.square(x - u), sizes.size -1, true)
    val y = (x - u) / AutoGrad.sqrt(s + e) // y: (-1, 2, 4) g2: (1, 4)
    y * weight + bias
  }

  def gelu(x: Variable[T]): Variable[T] = {
    x * 0.5 * (Activation("tanh").from((AutoGrad.square(x) * x * 0.044715 + x)
      * (scala.math.sqrt(2 / scala.math.Pi))) + 1)
  }

  def mlp(x: Variable[T]): Variable[T] = {
    val h = new Convolution1D(embeddingSize * 4, 1, init = RandomNormal(0.0, 0.02)).from(x)
    val a = gelu(h)
    val h2 = new Convolution1D(embeddingSize, 1, init = RandomNormal(0.0, 0.02)).from(a)
    Dropout(residPdrop).from(h2)
  }

  def attention(x: Variable[T]): Variable[T] = {
    val c = new Convolution1D(embeddingSize * 3, 1, init = RandomNormal(0.0, 0.02)).from(x)
    val query = c.slice(2, 0, embeddingSize)
    val key = c.slice(2, embeddingSize, embeddingSize)
    val value = c.slice(2, embeddingSize * 2, embeddingSize)
    val q = splitHeads(query)
    val k = splitHeads(key, k = true)
    val v = splitHeads(value)
    val a = attn(q, k, v, true) // m: (-1, 12, 77, 64)
    val m = mergeHeads(a) // m: (-1, 77, 768)
    val n = new Convolution1D(embeddingSize, 1, init = RandomNormal(0.0, 0.02))
      .from(m) // n: (-1, 77, 768)
    Dropout(residPdrop).from(n)
  }

  def splitHeads(x: Variable[T], k: Boolean = false): Variable[T] = {
    val sizes = x.getOutputShape().toSingle().toArray
    val newSizes = sizes.drop(1).dropRight(1) ++ Array(nHead, sizes.last / nHead)
    val r = Reshape(newSizes).from(x)
    if (k) Permute(Array(2, 3, 1)).from(r)
    else Permute(Array(2, 1, 3)).from(r)
  }

  def mergeHeads(x: Variable[T]): Variable[T] = {
    val p = AutoGrad.contiguous(Permute[T](Array(2, 1, 3)).from(x))
    val sizes = p.getOutputShape().toSingle().toArray
    Reshape(sizes.drop(1).dropRight(2) ++ Array(sizes.last * sizes(sizes.length - 2))).from(p)
  }

  // weights and ab belong to Attention
  val weights = Utils.tril(Tensor.ones(nCtx, nCtx)).view(1, nCtx, nCtx)
  val ab = Parameter[T](Shape(1, nCtx, nCtx), trainable = false, initWeight = weights)

  // scale shoule be set in Attention
  def attn(q: Variable[T], k: Variable[T], v: Variable[T], scale: Boolean = false): Variable[T] = {
    // q:(16, 12, 77, 64) k:(16, 12, 64, 77) v:(16, 12, 77, 64)
    var w = AutoGrad.mm(q, k) // w: (16, 12, 77, 77)
    if (scale) w = w / scala.math.sqrt(v.getOutputShape().toSingle().toArray.last)

    // mask attention
    w = w * ab + (ab * (-1) + 1) * -1e9
    w = Activation[Float]("softmax").from(w)
    w = Dropout(attnPdrop).from(w)

    AutoGrad.mm(w, v)
  }
}

object TransformerLayer {
  def apply[@specialized(Float, Double) T: ClassTag](
    vocab: Int = 40990,
    nCtx: Int = 77, // seq len
    embeddingSize: Int = 768,
    embeddingDrop: Double = 0.1,
    nLayer: Int = 12,
//    afn: String = "gelu",
    residPdrop: Double = 0.1,
    attnPdrop: Double = 0.1,
    nHead: Int = 12)(implicit ev: TensorNumeric[T]): TransformerLayer[T] = {
    new TransformerLayer[T](vocab, nCtx, embeddingSize, embeddingDrop, nLayer,
//      afn,
      residPdrop, attnPdrop, nHead)
  }
}

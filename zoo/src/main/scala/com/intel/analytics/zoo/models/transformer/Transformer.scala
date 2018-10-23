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

package com.intel.analytics.zoo.models.transformer

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Parameter, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.bigdl.tensor.Tensor

import scala.reflect.ClassTag

class Transformer[T: ClassTag] private(
  vocab: Int,
  nCtx: Int,
  embeddingSize: Int,
  embeddingDrop: Double,
  nLayer: Int,
  afn: String,
  residPdrop: Int,
  attnPdrop: Int,
  nHead: Int)(implicit ev: TensorNumeric[T])
  extends ZooModel[Activity, Activity, T] {

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    val input = Variable(inputShape = Shape(1, 2))
//    x = x.view(-1, x.size(-2), x.size(-1)) #(16, 77, 2)

    val e = Embedding(vocab, embeddingSize, inputLength = 77).from(input)

//  Add the position information to the input embeddings
    val h = AutoGrad.sum(e, 2, false)

    var nextInput: Variable[T] = h

    for (i <- 0 until nLayer) {
      val output = block(nextInput)
      nextInput = output
    }

    val model = Model(input, nextInput)
    model
  }

  def block(x: Variable[T]): Variable[T] = {
    val a = attention(x)
    val n = layerNorm(x + a)
    val m = mlp(n)
    val h = layerNorm2(n + m)
    h
  }

  // g, b for layerNorm
  val g = Parameter[T](Shape(embeddingSize), initWeight = Tensor.ones(embeddingSize))
  val b = Parameter[T](Shape(embeddingSize), initWeight = Tensor(embeddingSize))
  def layerNorm(x: Variable[T], e: Double = 1e-5): Variable[T] = {
    val u = AutoGrad.mean(x, -1, true)
    val s = AutoGrad.mean(AutoGrad.square(x - u), -1, true)
    val y = (x - u) / AutoGrad.sqrt(s + e)
    g * y + b
  }

  // g, b for layerNorm
  val g2 = Parameter[T](Shape(embeddingSize), initWeight = Tensor.ones(embeddingSize))
  val b2 = Parameter[T](Shape(embeddingSize), initWeight = Tensor(embeddingSize))
  def layerNorm2(x: Variable[T], e: Double = 1e-5): Variable[T] = {
    val u = AutoGrad.mean(x, -1, true)
    val s = AutoGrad.mean(AutoGrad.square(x - u), -1, true)
    val y = (x - u) / AutoGrad.sqrt(s + e)
    g2 * y + b2
  }

  def mlp(x: Variable[T]): Variable[T] = {
    val h = Conv1D(embeddingSize * 4, 1).from(x)
    val a = Activation(afn).from(h)
    val h2 = Conv1D(embeddingSize, 1).from(a)
    Dropout(residPdrop).from(h2)
  }

  def attention(x: Variable[T]): Variable[T] = {
    val c = Conv1D(embeddingSize * 3, 1).from(x)
    val query = c.slice(2, 0, embeddingSize)
    val key = c.slice(2, embeddingSize, embeddingSize)
    val value = c.slice(2, embeddingSize * 2, embeddingSize)
    val q = splitHeads(query)
    val k = splitHeads(key, k = true)
    val v = splitHeads(value)
    val a = attn(q, k, v)
    val m = mergeHeads(a)
    val n = Conv1D(embeddingSize, 1).from(m)
    Dropout(residPdrop).from(n)
  }

  def splitHeads(x: Variable[T], k: Boolean = false): Variable[T] = {
    val sizes = x.getOutputShape().toSingle().toArray
    val r = Reshape(sizes.dropRight(1) ++ Array(nHead, sizes.last)).from(x)
    if (k) Permute(Array(0, 2, 3, 1)).from(r)
    else Permute(Array(0, 2, 1, 3)).from(r)
  }

  def mergeHeads(x: Variable[T]): Variable[T] = {
    val p = AutoGrad.contiguous(Permute[T](Array(0, 2, 1, 3)).from(x))
    val sizes = p.getOutputShape().toSingle().toArray
    Reshape(sizes.dropRight(2) ++ Array(sizes.last * sizes(sizes.length - 2))).from(p)
  }

  // weights and ab belong to Attention
  val weights = Utils.tril(Tensor.ones(nCtx, nCtx)).view(1, 1, nCtx, nCtx)
  val ab = Parameter[T](Shape(1, 1, nCtx, nCtx), trainable = false, initWeight = weights)

  // scale shoule be set in Attention
  def attn(q: Variable[T], k: Variable[T], v: Variable[T], scale: Boolean = false): Variable[T] = {
    // TODO: should be matmul(q, k)
    var w = q * k
    if (scale) w = w / scala.math.sqrt(v.getOutputShape().toSingle().toArray.last)

    w = w * ab + (ab * (-1) + 1) * -1e9
    // TODO: softmax with last dim
    w = Activation("softmax").from(w)
    w = Dropout(attnPdrop).from(w)

    // TODO: should be matmul(w, v)
    w * v
  }
}

object Transformer {
  def apply[@specialized(Float, Double) T: ClassTag](
    vocab: Int,
    nCtx: Int,
    embeddingSize: Int,
    embeddingDrop: Double,
    nLayer: Int,
    afn: String,
    residPdrop: Int,
    attnPdrop: Int,
    nHead: Int)(implicit ev: TensorNumeric[T]): Transformer[T] = {
    new Transformer[T](vocab, nCtx, embeddingSize, embeddingDrop, nLayer,
      afn, residPdrop, attnPdrop, nHead).build()
  }

  /**
    * Load an existing Transformer model (with weights).
    *
    * @param path The path for the pre-defined model.
    *             Local file system, HDFS and Amazon S3 are supported.
    *             HDFS path should be like "hdfs://[host]:[port]/xxx".
    *             Amazon S3 path should be like "s3a://bucket/xxx".
    * @param weightPath The path for pre-trained weights if any. Default is null.
    * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
    */
  def loadModel[T: ClassTag](
    path: String,
    weightPath: String = null)(implicit ev: TensorNumeric[T]): Transformer[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[Transformer[T]]
  }
}

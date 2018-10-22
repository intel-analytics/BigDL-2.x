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

package com.intel.analytics.zoo.models.textranker

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.autograd.{Lambda, Variable, AutoGrad => A}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Kernel-pooling Neural Ranking Model with RBF kernel.
 *
 * @param text1Length
 * @param text2Length
 * @param vocabSize
 * @param embedSize
 * @param embedWeights
 * @param kernelNum
 * @param sigma
 * @param exactSigma
 */
class KNRM[T: ClassTag] private(
    val text1Length: Int,
    val text2Length: Int,
    val vocabSize: Int,
    val embedSize: Int,
    val embedWeights: Tensor[T] = null,
    val kernelNum: Int = 21,
    val sigma: Double = 0.1,
    val exactSigma: Double = 0.001)(implicit ev: TensorNumeric[T])
  extends ZooModel[Activity, Activity, T] {

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    // Remark: Share weights for embedding is not supported.
    // Thus here the model takes concatenated input and slice to split the input.
    val input = Input(inputShape = Shape(text1Length + text2Length))
    val embedding = Variable(Embedding(vocabSize, embedSize, weights = embedWeights)
      .inputs(input))
    val text1Embed = embedding.slice(1, 0, text1Length)
    val text2Embed = embedding.slice(1, text1Length, text2Length)
    val mm = A.batchDot(text1Embed, text2Embed, axes = List(2, 2)) // Translation Matrix.
    val KM = new ArrayBuffer[Variable[T]]()
    for (i <- 0 until kernelNum) {
      var mu = 1.0 / (kernelNum - 1) + (2.0 * i) / (kernelNum - 1) - 1.0
      val _sigma = if (mu > 1.0) {
        mu = 1.0
        exactSigma
      } else sigma
      val mmExp = A.exp[T]((mm - mu) * (mm - mu) / _sigma / _sigma * (-0.5))
      val mmDocSum = A.sum(mmExp, axis = 2)
      val mmLog = A.log(mmDocSum + 1.0)
      // Remark: Keep the reduced dimension for the last sum and squeeze after stack.
      // Otherwise, when batch=1, the output will become a Scalar not compatible stack.
      val mmSum = A.sum(mmLog, 1, keepDims = true)
      KM.append(mmSum)
    }
    val Phi = Squeeze(2).inputs(A.stack(KM.toList).node)
    val output = Dense(1, init = "uniform", activation = "sigmoid").inputs(Phi)
    Model(input, output)
  }
}

object KNRM {
  def apply[@specialized(Float, Double) T: ClassTag](
      text1Length: Int,
      text2Length: Int,
      vocabSize: Int,
      embedSize: Int,
      embedWeights: Tensor[T] = null,
      kernelNum: Int = 21,
      sigma: Double = 0.1,
      exactSigma: Double = 0.001)(implicit ev: TensorNumeric[T]): KNRM[T] = {
    new KNRM[T](text1Length, text2Length, vocabSize, embedSize, embedWeights,
      kernelNum, sigma, exactSigma).build()
  }
}

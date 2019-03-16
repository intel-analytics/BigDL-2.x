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

import com.intel.analytics.bigdl.nn.{BatchNormalization => BBatchNormalization}
import com.intel.analytics.bigdl.nn.{Xavier, RandomUniform, RandomNormal}
import com.intel.analytics.bigdl.nn.keras.{BatchNormalization => BKBatchNormalization}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Batch normalization layer.
 * Normalize the activations of the previous layer at each batch,
 * i.e. applies a transformation that maintains the mean activation
 * close to 0 and the activation standard deviation close to 1.
 * It is a feature-wise normalization, each feature map in the input will be normalized separately.
 * The input of this layer should be 4D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param epsilon Small Double > 0. Fuzz parameter. Default is 0.001.
 * @param momentum Double. Momentum in the computation of the exponential average
 *                 of the mean and standard deviation of the data,
 *                 for feature-wise normalization. Default is 0.99.
 * @param betaInit Name of initialization function for shift parameter. Default is 'zero'.
 * @param gammaInit Name of initialization function for scale parameter. Default is 'one'.
 * @param dimOrdering Format of input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 *                    For NCHW, axis along which to normalize is 1. For NHWC, axis is 3.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class BatchNormalization[T: ClassTag](
   override val epsilon: Double = 0.001,
   override val momentum: Double = 0.99,
   override val betaInit: String = "zero",
   override val gammaInit: String = "one",
   override val dimOrdering: DataFormat = DataFormat.NCHW,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BKBatchNormalization[T](
    epsilon, momentum, betaInit, gammaInit, dimOrdering, inputShape) with Net {

  private def getInit(init: String, n: Int): Tensor[T] = {
    val weights = Tensor[T](n)
    init.toLowerCase() match {
      case "zero" => weights.fill(ev.zero)
      case "one" => weights.fill(ev.one)
      case "glorot_uniform" => Xavier.init(weights)
        weights
      case "uniform" => RandomUniform(-0.05, 0.05).init(weights)
        weights
      case "normal" => RandomNormal(0.0, 0.05).init(weights)
        weights
      case _ => throw new IllegalArgumentException(s"Unsupported initialization method: " +
        s"${init.toLowerCase()}")
    }
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4 || input.length == 2,
      s"BatchNormalization requires 4D or 2D input, but got input dim ${input.length}")
    inputShape
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    if(input.length == 4) {
     super.doBuild(inputShape)
    } else {
      val nOutput = input(1)
      BBatchNormalization[T](nOutput,
        epsilon,
        momentum,
        affine = true,
        initWeight = getInit(gammaInit, nOutput),
        initBias = getInit(betaInit, nOutput))
    }
  }
}

object BatchNormalization {
  def apply[@specialized(Float, Double) T: ClassTag](
    epsilon: Double = 0.001,
    momentum: Double = 0.99,
    betaInit: String = "zero",
    gammaInit: String = "one",
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): BatchNormalization[T] = {
    new BatchNormalization[T](epsilon, momentum, betaInit, gammaInit,
      KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}

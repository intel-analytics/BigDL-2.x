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

package com.intel.analytics.zoo.pipeline.api.keras2.layers

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable}
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, Dense => BigDLDense}
import com.intel.analytics.bigdl.nn.{Container, InitializationMethod, Xavier, Zeros}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}

import scala.reflect.ClassTag

/**
 * A densely-connected NN layer.
 * The most common input is 2D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param units The size of output dimension.
 * @param kernelInitializer Initialization method for the weights of the layer. Default is Xavier.
 *             You can also pass in corresponding string representations such as 'glorot_uniform'
 *             or 'normal', etc. for simple init methods in the factory method.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param kernelRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param biasRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param useBias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Dense[T: ClassTag](
    val units: Int,
    val kernelInitializer: InitializationMethod = Xavier,
    val biasInitializer: InitializationMethod = Zeros,
    override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
    val kernelRegularizer: Regularizer[T] = null,
    val biasRegularizer: Regularizer[T] = null,
    val useBias: Boolean = true,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends klayers1.Dense[T](outputDim = units,
    init = kernelInitializer,
    activation = activation,
    wRegularizer = kernelRegularizer,
    bRegularizer = biasRegularizer,
    bias = useBias,
    inputShape = inputShape) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val module = super.doBuild(inputShape)
    Net.setInitMethod(module,
      weightInitMethod = kernelInitializer, biasInitMethod = biasInitializer)
    module.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Dense {

  def apply[@specialized(Float, Double) T: ClassTag] (
      units: Int,
      kernelInitializer: String = "glorot_uniform",
      biasInitializer: String = "zero",
      activation: String = null,
      kernelRegularizer: Regularizer[T] = null,
      biasRegularizer: Regularizer[T] = null,
      useBias: Boolean = true,
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Dense[T] = {
    new Dense[T](
      units = units,
      kernelInitializer = KerasUtils.getInitMethod(kernelInitializer),
      biasInitializer = KerasUtils.getInitMethod(biasInitializer),
      activation = KerasUtils.getKerasActivation[T](activation),
      kernelRegularizer = kernelRegularizer,
      biasRegularizer = biasRegularizer,
      useBias = useBias,
      inputShape = inputShape)
  }
}



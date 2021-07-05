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

import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}
import scala.reflect.ClassTag

/**
 * Locally-connected layer for 1D inputs which works similarly to the TemporalConvolution layer,
 * except that weights are unshared, that is, a different set of filters
 * is applied at each different patch of the input.
 * Padding currently supported for this layer is 'valid'.
 * The input of this layer should be 3D.
 *
 * When using this layer as the first layer in a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param filters Dimensionality of the output.
 * @param kernelSize The extension (spatial or temporal) of each filter.
 * @param strides Integer. Factor by which to subsample output.
 * @param padding Only 'valid is supported for now.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param kernelRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param biasRegularizer An instance of [[Regularizer]], applied to the bias. Default is null.
 * @param useBias Whether to include a bias (i.e. make the layer affine rather than linear).
 *             Default is true.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class LocallyConnected1D[T: ClassTag](
      val filters: Int,
      val kernelSize: Int,
      val strides: Int = 1,
      val padding: String = "valid",
      override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
      val kernelRegularizer: Regularizer[T] = null,
      val biasRegularizer: Regularizer[T] = null,
      val useBias: Boolean = true,
      override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends klayers1.LocallyConnected1D[T](nbFilter = filters, filterLength = kernelSize,
    activation, subsampleLength = strides, wRegularizer = kernelRegularizer,
    bRegularizer = biasRegularizer, bias = useBias, inputShape) with Net {}

object LocallyConnected1D {
  def apply[@specialized(Float, Double) T: ClassTag](
      filters: Int,
      kernelSize: Int,
      strides: Int = 1,
      padding: String = "valid",
      activation: String = null,
      kernelRegularizer: Regularizer[T] = null,
      biasRegularizer: Regularizer[T] = null,
      useBias: Boolean = true,
       inputShape: Shape = null)(implicit ev: TensorNumeric[T]): LocallyConnected1D[T] = {
    new LocallyConnected1D[T](filters, kernelSize, strides, padding,
      KerasUtils.getKerasActivation(activation),
      kernelRegularizer, biasRegularizer, useBias, inputShape)
  }
}


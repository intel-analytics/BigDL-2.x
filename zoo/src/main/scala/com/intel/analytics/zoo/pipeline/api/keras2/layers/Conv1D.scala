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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.nn.{InitializationMethod, Xavier, Zeros}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}

import scala.reflect.ClassTag

/**
 * 1D convolution layer (e.g. temporal convolution).
 * This layer creates a convolution kernel that is convolved
 * with the layer input over a single spatial (or temporal) dimension
 * to produce a tensor of outputs.
 * If `use_bias` is True, a bias vector is created and added to the outputs.
 * Finally, if `activation` is not `None`,
 * it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide an `input_shape` argument
 * (tuple of integers or `None`, e.g.
 * `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
 * or `(None, 128)` for variable-length sequences of 128-dimensional vectors.
 *
 * Input shape
 * 3D tensor with shape: `(batch_size, steps, input_dim)`
 *
 * Output shape
 * 3D tensor with shape: `(batch_size, new_steps, filters)`
 * `steps` value might have changed due to padding or strides.
 *
 * @param filters Integer, the dimensionality of the output space
 *                (i.e. the number of output filters in the convolution).
 * @param kernelSize: An integer or tuple/list of a single integer,
 *                specifying the length of the 1D convolution window.
 * @param strides: An integer or tuple/list of a single integer,
 *               specifying the stride length of the convolution.
 *               Specifying any stride value != 1 is incompatible with specifying
 *               any `dilation_rate` value != 1.
 * @param padding One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
 *                `"valid"` means "no padding".
 *                `"same"` results in padding the input such that
 *                the output has the same length as the original input.
 *                `"causal"` results in causal (dilated) convolutions, e.g. output[t]
 *                does not depend on input[t+1:]. Useful when modeling temporal data
 *                where the model should not violate the temporal order.
 * @param activation Activation function to use. Default is null.
 *                   You can also pass in corresponding string representations such as 'relu'
 *                   or 'sigmoid', etc. for simple activations in the factory method.
 * @param useBias Whether to include a bias (i.e. make the layer affine rather than linear).
 *                 Default is true.
 * @param kernelInitializer Initializer for the `kernel` weights matrix.
 * @param biasInitializer Initializer for the bias vector.
 * @param kernelRegularizer Regularizer function applied to
 *                           the `kernel` weights matrix Default is null.
 * @param biasRegularizer Regularizer function applied to the bias vector. Default is null.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class Conv1D[T: ClassTag](
   val filters: Int,
   val kernelSize: Int,
   val strides: Int = 1,
   val padding: String = "valid",
   override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val useBias: Boolean = true,
   val kernelInitializer: InitializationMethod = Xavier,
   val biasInitializer: InitializationMethod = Zeros,
   val kernelRegularizer: Regularizer[T] = null,
   val biasRegularizer: Regularizer[T] = null,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends klayers1.Convolution1D[T](nbFilter = filters, filterLength = kernelSize,
    init = kernelInitializer, activation = activation, borderMode = padding,
    subsampleLength = strides, wRegularizer = kernelRegularizer,
    bRegularizer = biasRegularizer, bias = useBias, inputShape = inputShape) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val module = super.doBuild(inputShape)
    Net.setInitMethod(module,
      weightInitMethod = kernelInitializer, biasInitMethod = biasInitializer)
    module.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Conv1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    filters: Int,
    kernelSize: Int,
    strides: Int = 1,
    padding: String = "valid",
    activation: String = null,
    useBias: Boolean = true,
    kernelInitializer: String = "glorot_uniform",
    biasInitializer: String = "zero",
    kernelRegularizer: Regularizer[T] = null,
    biasRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)
    (implicit ev: TensorNumeric[T]): Conv1D[T] = {
    val kernelInitValue = KerasUtils.getInitMethod(kernelInitializer)
    val biasInitValue = KerasUtils.getInitMethod(biasInitializer)
    val activationValue = KerasUtils.getKerasActivation(activation)
    new Conv1D[T](
      filters, kernelSize, strides, padding, activationValue,
      useBias, kernelInitValue, biasInitValue, kernelRegularizer,
      biasRegularizer, inputShape)
  }
}

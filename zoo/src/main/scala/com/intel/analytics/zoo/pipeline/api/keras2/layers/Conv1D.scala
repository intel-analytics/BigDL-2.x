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
  * Applies convolution operator for filtering neighborhoods of 1-D inputs.
  * You can also use Conv1D as an alias of this layer.
  * The input of this layer should be 3D.
  *
  * When you use this layer as the first layer of a model, you need to provide the argument
  * inputShape (a Single Shape, does not include the batch dimension).
  *
  * @param filters Integer, the dimensionality of the output space
  *                (i.e. the number of output filters in the convolution).
  * @param kernel_size: An integer or tuple/list of a single integer,
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
  *                See [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499).
  * @param data_format: A string,
  *                   one of `"channels_last"` (default) or `"channels_first"`.
  *                   The ordering of the dimensions in the inputs.
  *                   `"channels_last"` corresponds to inputs with shape
  *                   `(batch, length, channels)`
  *                   (default format for temporal data in Keras)
  *                   while `"channels_first"` corresponds to inputs
  *                   with shape `(batch, channels, length)`.
  * @param activation Activation function to use. Default is null.
  *                   You can also pass in corresponding string representations such as 'relu'
  *                   or 'sigmoid', etc. for simple activations in the factory method.
  * @param use_bias Whether to include a bias (i.e. make the layer affine rather than linear).
  *                 Default is true.
  * @param kernel_initializer Initializer for the `kernel` weights matrix.
  * @param bias_initializer Initializer for the bias vector.
  * @param kernel_Regularizer Regularizer function applied to
  *                           the `kernel` weights matrix Default is null.
  * @param bias_Regularizer Regularizer function applied to the bias vector. Default is null.
  * @param activity_Regularizer Regularizer function applied to
  *                             the output of the layer (its "activation").
  * @param inputShape A Single Shape, does not include the batch dimension.
  * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
  */
class Conv1D[T: ClassTag](
                         val filters: Int,
                         val kernel_size: Int,
                         val strides: Int = 1,
                         val padding: String = "valid",
                         val data_format: String = "channels_last",
                         override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
                         val use_bias: Boolean = true,
                         val kernel_initializer: InitializationMethod = Xavier,
                         val bias_initializer: InitializationMethod = Zeros,
                         val kernel_Regularizer: Regularizer[T] = null,
                         val bias_Regularizer: Regularizer[T] = null,
                         val activity_Regularizer: Regularizer[T] = null,
  override val inputShape: Shape = null)
  (implicit ev: TensorNumeric[T])
  extends klayers1.Convolution1D[T](nbFilter = filters, filterLength = kernel_size, init = kernel_initializer,
    activation = activation, borderMode = padding, subsampleLength = strides, wRegularizer = kernel_Regularizer,
    bRegularizer = bias_Regularizer, bias = use_bias, inputShape = inputShape) with Net {}

object Conv1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    filters: Int,
    kernel_size: Int,
    strides: Int = 1,
    padding: String = "valid",
    data_format: String = "channels_last",
    activation: String = null,
    use_bias: Boolean = true,
    kernel_Initializer: String = "glorot_uniform",
    bias_Initializer: String = "zero",
    kernel_Regularizer: Regularizer[T] = null,
    bias_Regularizer: Regularizer[T] = null,
    activity_Regularizer: Regularizer[T] = null,
    inputShape: Shape = null)
    (implicit ev: TensorNumeric[T]): Conv1D[T] = {
    val kernelInitValue = KerasUtils.getInitMethod(kernel_Initializer)
    val biasInitValue = KerasUtils.getInitMethod(bias_Initializer)
    val activationValue = KerasUtils.getKerasActivation(activation)
    new Conv1D[T](
      filters, kernel_size, strides, padding, data_format,
      activationValue, use_bias, kernelInitValue, biasInitValue, kernel_Regularizer,
      bias_Regularizer, activity_Regularizer, inputShape)
  }
}
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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.nn.{InitializationMethod, Xavier, Zeros}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.{KerasUtils => KerasUtils1}
import com.intel.analytics.zoo.pipeline.api.keras2.layers.utils.{KerasUtils => KerasUtils2}
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}

import scala.reflect.ClassTag

/**
 * 2D convolution layer (e.g. spatial convolution over images).
 *
 * This layer creates a convolution kernel that is convolved
 * with the layer input to produce a tensor of
 * outputs. If `use_bias` is True,
 * a bias vector is created and added to the outputs. Finally, if
 * `activation` is not `None`, it is applied to the outputs as well.
 *
 * When using this layer as the first layer in a model,
 * provide the keyword argument `input_shape`
 * (tuple of integers, does not include the sample axis),
 * e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
 * in `data_format="channels_last"`.
 *
 * Input shape
 * 4D tensor with shape:
 * `(samples, channels, rows, cols)` if data_format='channels_first'
 * or 4D tensor with shape:
 * `(samples, rows, cols, channels)` if data_format='channels_last'.
 *
 * Output shape
 * 4D tensor with shape:
 * `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
 * or 4D tensor with shape:
 * `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
 * `rows` and `cols` values might have changed due to padding.
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
 * @param dataFormat A string,
 *                   one of `channels_last` (default) or `channels_first`.
 *                   The ordering of the dimensions in the inputs.
 *                   `channels_last` corresponds to inputs with shape
 *                   `(batch, height, width, channels)` while `channels_first`
 *                   corresponds to inputs with shape
 *                   `(batch, channels, height, width)`.
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
class Conv2D[T: ClassTag](
   val filters: Int,
   val kernelSize: Array[Int],
   val strides: Array[Int] = Array(1, 1),
   val padding: String = "valid",
   val dataFormat: DataFormat = DataFormat.NCHW,
   override val activation: KerasLayer[Tensor[T], Tensor[T], T] = null,
   val useBias: Boolean = true,
   val kernelInitializer: InitializationMethod = Xavier,
   val biasInitializer: InitializationMethod = Zeros,
   val kernelRegularizer: Regularizer[T] = null,
   val biasRegularizer: Regularizer[T] = null,
   override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends klayers1.Convolution2D[T](nbFilter = filters, nbRow = kernelSize(0),
    nbCol = kernelSize(1), init = kernelInitializer, activation = activation,
    borderMode = padding, dimOrdering = dataFormat,
    subsample = strides, wRegularizer = kernelRegularizer,
    bRegularizer = biasRegularizer, bias = useBias, inputShape = inputShape) with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val module = super.doBuild(inputShape)
    Net.setInitMethod(module,
      weightInitMethod = kernelInitializer, biasInitMethod = biasInitializer)
    module.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Conv2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    filters: Int,
    kernelSize: Array[Int],
    strides: Array[Int] = Array(1, 1),
    padding: String = "valid",
    dataFormat: String = "channels_first",
    activation: String = null,
    useBias: Boolean = true,
    kernelInitializer: String = "glorot_uniform",
    biasInitializer: String = "zero",
    kernelRegularizer: Regularizer[T] = null,
    biasRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)
  (implicit ev: TensorNumeric[T]): Conv2D[T] = {
    val kernelInitValue = KerasUtils1.getInitMethod(kernelInitializer)
    val biasInitValue = KerasUtils1.getInitMethod(biasInitializer)
    val activationValue = KerasUtils1.getKerasActivation(activation)
    val dataFormatValue = KerasUtils2.toBigDLFormat(dataFormat)
    new Conv2D[T](
      filters, kernelSize, strides, padding, dataFormatValue, activationValue,
      useBias, kernelInitValue, biasInitValue, kernelRegularizer,
      biasRegularizer, inputShape)
  }
}

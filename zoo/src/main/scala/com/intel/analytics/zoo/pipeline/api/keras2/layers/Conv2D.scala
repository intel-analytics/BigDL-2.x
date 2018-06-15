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

import com.intel.analytics.bigdl.nn.{InitializationMethod, Xavier}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.Convolution2D
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}

import scala.reflect.ClassTag

/**
* Applies a 2D convolution over an input image composed of several input planes.
* You can also use Conv2D as an alias of this layer.
* The input of this layer should be 4D.
*
* When you use this layer as the first layer of a model, you need to provide the argument
* inputShape (a Single Shape, does not include the batch dimension),
* e.g. inputShape=Shape(3, 128, 128) for 128x128 RGB pictures.
*
* @param filters: Integer, the dimensionality of the output space
*                 (i.e. the number of output filters in the convolution).
* @param kernel_size: An integer or tuple/list of a single integer,
*                 specifying the length of the 1D convolution window.
* @param strides: An integer or tuple/list of a single integer,
*                 specifying the stride length of the convolution.
*                 Specifying any stride value != 1 is incompatible with specifying
*                 any `dilation_rate` value != 1.
* @param  padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
*`                 "valid"` means "no padding". "same" results in padding the input such that
*                  the output has the same length as the original input.
*`                 "causal"` results in causal (dilated) convolutions, e.g. output[t]
*                  does not depend on input[t+1:]. Useful when modeling temporal data
*                  where the model should not violate the temporal order.
*                  See [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499).
* @param data_format: A string,
*                     one of `channels_last` (default) or `channels_first`.
*                     The ordering of the dimensions in the inputs.
*                  `channels_last` corresponds to inputs with shape
*            `(batch, height, width, channels)` while `channels_first`
*            corresponds to inputs with shape
*            `(batch, channels, height, width)`.
*            It defaults to the `image_data_format` value found in your
*            Keras config file at `~/.keras/keras.json`.
*            If you never set it, then it will be "channels_last".
* @param dilation_rate: an integer or tuple/list of a single integer, specifying
*                       the dilation rate to use for dilated convolution.
*                       Currently, specifying any `dilation_rate` value != 1 is
*                        incompatible with specifying any `strides` value != 1.
* @param activation: Activation function to use
*                   (see [activations](../activations.md)).
*                   If you don't specify anything, no activation is applied
*                   (ie. "linear" activation: `a(x) = x`).
* @param use_bias: Boolean, whether the layer uses a bias vector.
* @param kernel_initializer: Initializer for the `kernel` weights matrix
*                            (see [initializers](../initializers.md)).
* @param bias_initializer: Initializer for the bias vector
                           (see [initializers](../initializers.md)).
* @param kernel_regularizer: Regularizer function applied to
*                            the `kernel` weights matrix
*                            (see [regularizer](../regularizers.md)).
* @param bias_regularizer: Regularizer function applied to the bias vector
                          (see [regularizer](../regularizers.md)).
* @param activity_regularizer: Regularizer function applied to
*                              the output of the layer (its "activation").
                               (see [regularizer](../regularizers.md)).
* @param kernel_constraint: Constraint function applied to the kernel matrix
*                           (see [constraints](../constraints.md)).
* @parambias_constraint: Constraint function applied to the bias vector
*                       see [constraints](../constraints.md)).
*/

class Conv2D[T: ClassTag](
                         val filters :Int,
                         val kernelSize : Int,
                         val strides : Init = 1,
                         val padding : String = "valid",
                         /*val dataFormat*/
                         /*val dilationRate,*/
                         override val activation: KerasLayer[Tensor[T],Tensor[T],T]= null,
                         val useBias : Boolean =True,
                         val kernelRegularizer : InitializationMethod = Xavier,
                         val biasRegularizer : Regularizer[T] = null,
                         /*val activityRegularizer*/
                         override val inputShape:Shape = null)
                         (implicit ev:TensorNumeric[T])
                          extends klayers1.Convolution1D[T](nbFilter = filters,filterLength = kernelSize,init = kernelInitializer,activation = avtivation,borderMore = padding,subsampleLength = strides,wRegularizer = kernelRegularize,
                            bRegularizer = biasRegularizer,bias = useBias,inputShape =inputShape)with Net{}


object Conv2D{
  def apply[@specialized(Float,Double)T:ClassTag](
                                                 filters : Int,
                                                 kernelSize : Int,
                                                 strides : Init =1,
                                                 padding :String ="valid",
                                                 activation : String = "channels_last",
                                                 useBias : Boolean = true,
                                                 kernelInitializer : String = "glorot_uniform",
                                                 biasRegularizer : Regularizer[T],
                                                 inputShape : Shape = null)
                                                 (implicit ev: TensorNumeric[T]): Conv2D[T] = {
                                                   val kernelInitValue = KerasUtils.getInitMethod(kernelInitializer)
                                                   val biasInitValue = KerasUtils.getInitMethod(biasInitializer)
                                                   val activationValue = KerasUtils.getKerasActivation(action)


                                                 new Conv2D[T](filters,kernelSize, strides, padding,activation,
                                                 useBias,kernelInitializer,biasRegularizer, kernelValue,biasIitValue,actionValue,inputShape
                                                  )
                                                  }

}

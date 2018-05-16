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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, IdentityOutputShape}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * This layer has a weight with given size.
 * The weight will be multiplied element-wise to the input.
 * If the element number of the weight matches the input,
 * a simple element-wise multiplication will be done.
 * Or the bias will be expanded to the same size of the input.
 * The expand means repeat on unmatched singleton dimension (if some unmatched dimension isn't
 * singleton dimension, an error will be raised).
 *
 * When you use this layer as the first layer of a model, you need to provide
 * the argument inputShape (a Single Shape, does not include the batch dimension).
 *
 * Remark: This layer is from Torch and wrapped in Keras style.
 *
 * @param size The size of the bias.
 * @param wRegularizer An instance of [[Regularizer]], (eg. L1 or L2 regularization),
 *                     applied to the input weights matrices. Default is null.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class CMul[T: ClassTag](
    val size: Array[Int],
    var wRegularizer: Regularizer[T] = null,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))
    with IdentityOutputShape with Net {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.CMul(size, wRegularizer)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object CMul {
  def apply[@specialized(Float, Double) T: ClassTag](
    size: Array[Int],
    wRegularizer: Regularizer[T] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): CMul[T] = {
    new CMul[T](size, wRegularizer, inputShape)
  }
}

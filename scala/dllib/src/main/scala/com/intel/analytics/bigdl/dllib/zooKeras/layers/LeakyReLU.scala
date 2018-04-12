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

import com.intel.analytics.bigdl.nn.keras.{LeakyReLU => BigDLLeakyReLU}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Leaky version of a Rectified Linear Unit.
 * It allows a small gradient when the unit is not active:
 * f(x) = alpha * x for x < 0,
 * f(x) = x for x >= 0.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param alpha Double >= 0. Negative slope coefficient. Default is 0.3.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class LeakyReLU[T: ClassTag](
    private val alpha: Double = 0.3,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLLeakyReLU[T](
    alpha, inputShape) {
}

object LeakyReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
    alpha: Double = 0.3,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): LeakyReLU[T] = {
    new LeakyReLU[T](alpha, inputShape)
  }
}

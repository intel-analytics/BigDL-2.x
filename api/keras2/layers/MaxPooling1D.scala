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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}

import scala.reflect.ClassTag

/**
 * Max pooling operation for temporal data.
 *
 * Input shape
 * 3D tensor with shape: `(batch_size, steps, features)`.
 *
 * Output shape
 * 3D tensor with shape: `(batch_size, downsampled_steps, features)`.
 *
 * @param poolSize Size of the region to which max pooling is applied. Integer. Default is 2.
 * @param strides Factor by which to downscale. Integer, or -1. 2 will halve the input.
 *               If -1, it will default to poolSize. Default is -1.
 * @param padding One of `"valid"` or `"same"` (case-insensitive). Default is 'valid'.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
class MaxPooling1D[T: ClassTag](
                                 val poolSize: Int = 2,
                                 val strides: Int = -1,
                                 val padding: String = "valid",
                                 override val inputShape: Shape = null)(
  implicit ev: TensorNumeric[T])
  extends klayers1.MaxPooling1D[T](poolLength = poolSize,
    stride = strides, borderMode = padding, inputShape = inputShape) with Net {

}

object MaxPooling1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    poolSize: Int = 2,
    strides: Int = -1,
    padding: String = "valid",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): MaxPooling1D[T] = {
    new MaxPooling1D[T](poolSize, strides, padding, inputShape)
  }
}

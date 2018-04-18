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

import com.intel.analytics.bigdl.nn.keras.{GlobalMaxPooling3D => BigDLGlobalMaxPooling3D}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies global max pooling operation for 3D data.
 * Data format currently supported for this layer is 'CHANNEL_FIRST' (dimOrdering='th').
 * Border mode currently supported for this layer is 'valid'.
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param dimOrdering Format of input data. Please use 'CHANNEL_FIRST' (dimOrdering='th').
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class GlobalMaxPooling3D[T: ClassTag](
   dimOrdering: String = "CHANNEL_FIRST",
   inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLGlobalMaxPooling3D[T](dimOrdering, inputShape) with Net {}

object GlobalMaxPooling3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : GlobalMaxPooling3D[T] = {
    new GlobalMaxPooling3D[T](KerasUtils.toBigDLFormat5D(dimOrdering), inputShape)
  }
}

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

import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.nn.VolumetricMaxPooling
import com.intel.analytics.bigdl.nn.{Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras2.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}
import scala.reflect.ClassTag

/**
 * Applies global max pooling operation for 3D data.
 * Data format currently supported for this layer is 'channels_first' (.
 * padding currently supported for this layer is 'valid'.
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param dataFormat Format of input data. Please use 'channels_first' .
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class GlobalMaxPooling3D[T: ClassTag](
      val dataFormat: String = "channels_first",
       override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends klayers1.GlobalMaxPooling3D[T](dimOrdering = dataFormat, inputShape) with Net {

}

object GlobalMaxPooling3D {
  def apply[@specialized(Float, Double) T: ClassTag](
      dataFormat: String = "channels_first",
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]) : GlobalMaxPooling3D[T] = {
    new GlobalMaxPooling3D[T](KerasUtils.toBigDLFormat5D(dataFormat), inputShape)
  }
}


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

import com.intel.analytics.bigdl.nn.{SpatialAveragePooling, Sequential => TSequential}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.keras.GlobalPooling2D
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras2.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}
import scala.reflect.ClassTag


/**
 * Applies global average pooling operation for spatial data.
 * The input of this layer should be 4D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param dataFormat Format of input data. Please use DataFormat.NCHW (dataFormat='channels_first')
 *                    or DataFormat.NHWC (dataFormat='channels_last'). Default is NCHW.
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T The numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class GlobalAveragePooling2D[T: ClassTag](
      val dataFormat: DataFormat = DataFormat.NCHW,
      override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends klayers1.GlobalAveragePooling2D[T](dataFormat, inputShape) with Net {

}

object GlobalAveragePooling2D {
  def apply[@specialized(Float, Double) T: ClassTag](
      dataFormat: String = "channels_first",
       inputShape: Shape = null)(implicit ev: TensorNumeric[T]): GlobalAveragePooling2D[T] = {
    new GlobalAveragePooling2D[T](KerasUtils.toBigDLFormat(dataFormat), inputShape)
  }
}

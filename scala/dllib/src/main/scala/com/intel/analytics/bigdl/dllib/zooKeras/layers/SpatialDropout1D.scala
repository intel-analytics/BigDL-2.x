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

import com.intel.analytics.bigdl.nn.keras.{SpatialDropout1D => BigDLSpatialDropout1D}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * Spatial 1D version of Dropout.
 * This version performs the same function as Dropout, however it drops
 * entire 1D feature maps instead of individual elements. If adjacent frames
 * within feature maps are strongly correlated (as is normally the case in
 * early convolution layers) then regular dropout will not regularize the
 * activations and will otherwise just result in an effective learning rate
 * decrease. In this case, SpatialDropout1D will help promote independence
 * between feature maps and should be used instead.
 * The input of this layer should be 3D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param p Fraction of the input units to drop. Double between 0 and 1.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class SpatialDropout1D[T: ClassTag](
    override val p: Double = 0.5,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLSpatialDropout1D[T](
    p, inputShape) {
}

object SpatialDropout1D {
  def apply[@specialized(Float, Double) T: ClassTag](
    p: Double = 0.5,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SpatialDropout1D[T] = {
    new SpatialDropout1D[T](p, inputShape)
  }
}

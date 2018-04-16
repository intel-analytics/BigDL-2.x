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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Applies max pooling operation for spatial data.
 * The input of this layer should be 4D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param poolSize Int array of length 2 corresponding to the downscale vertically and
 *                 horizontally. Default is (2, 2), which will halve the image in each dimension.
 * @param strides Int array of length 2. Stride values. Default is null, and in this case it will
 *                be equal to poolSize.
 * @param borderMode Either 'valid' or 'same'. Default is 'valid'.
 * @param dimOrdering Format of input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class MaxPooling2D[T: ClassTag](
  poolSize: Array[Int] = Array(2, 2),
  strides: Array[Int] = null,
  borderMode: String = "valid",
  dimOrdering: DataFormat = DataFormat.NCHW,
  inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends com.intel.analytics.bigdl.nn.keras.MaxPooling2D[T](poolSize, strides, borderMode,
    dimOrdering, inputShape) with Net {}

object MaxPooling2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    poolSize: (Int, Int) = (2, 2),
    strides: (Int, Int) = null,
    borderMode: String = "valid",
    dimOrdering: String = "th",
    inputShape: Shape = null)
    (implicit ev: TensorNumeric[T]): MaxPooling2D[T] = {
    val poolSizeArray = poolSize match {
      case null => throw new IllegalArgumentException("For MaxPooling2D, " +
        "poolSize can not be null, please input int tuple of length 2")
      case _ => Array(poolSize._1, poolSize._2)
    }
    val stridesArray = strides match {
      case null => null
      case _ => Array(strides._1, strides._2)
    }
    val dimOrderingValue = KerasUtils.toBigDLFormat(dimOrdering)
    new MaxPooling2D[T](poolSizeArray, stridesArray, borderMode, dimOrderingValue, inputShape)
  }
}

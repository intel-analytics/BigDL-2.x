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
import com.intel.analytics.zoo.pipeline.api.keras2.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}
import scala.reflect.ClassTag

/**
 * Applies max pooling operation for 3D data (spatial or spatio-temporal).
 * Data format currently supported for this layer is 'channels_first'.
 * Padding currently supported for this layer is 'valid'.
 * The input of this layer should be 5D.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param poolSize Int array of length 3. Factors by which to downscale (dim1, dim2, dim3).
 *                 Default is (2, 2, 2), which will halve the image in each dimension.
 * @param strides Int array of length 3. Stride values. Default is null, and in this case it will
 *                be equal to poolSize.
 * @param padding  Only 'valid' is supported new .
 * @param dataFormat Format of input data. Please use "channels_first".
 * @param inputShape A Single Shape, does not include the batch dimension.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 */
class MaxPooling3D[T: ClassTag](
      override val poolSize: Array[Int] = Array(2, 2, 2),
      override val strides: Array[Int] = null,
      val padding: String = "valid",
      val dataFormat: String = "channels_first",
      override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends klayers1.MaxPooling3D[T](
    poolSize = poolSize, strides = strides, dimOrdering = dataFormat,
    inputShape = inputShape) with Net {
}

object MaxPooling3D {
  def apply[@specialized(Float, Double) T: ClassTag](
      poolSize: (Int, Int, Int) = (2, 2, 2),
      strides: (Int, Int, Int) = null,
      padding: String = "valid",
      dataFormat: String = "channels_first",
      inputShape: Shape = null)(implicit ev: TensorNumeric[T]): MaxPooling3D[T] = {
    val poolSizeArray = poolSize match {
      case null => throw new IllegalArgumentException("For MaxPooling3D, " +
        "poolSize can not be null, please input int tuple of length 3")
      case _ => Array(poolSize._1, poolSize._2, poolSize._3)
    }
    val strideArray = strides match {
      case null => null
      case _ => Array(strides._1, strides._2, strides._3)
    }
    new MaxPooling3D[T](poolSizeArray, strideArray, padding,
      KerasUtils.toBigDLFormat5D(dataFormat), inputShape)
  }
}



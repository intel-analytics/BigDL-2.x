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

import com.intel.analytics.bigdl.nn.SpatialCrossMapLRN
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat, IdentityOutputShape}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Local Response Normalization between different feature maps.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param alpha Double. The scaling parameter. Default is 0.0001.
 * @param k Double. A constant.
 * @param beta Double. The exponent. Default is 0.75.
 * @param n The number of channels to sum over.
 * @param dimOrdering Format of input data. Either DataFormat.NCHW (dimOrdering='th') or
 *                    DataFormat.NHWC (dimOrdering='tf'). Default is NCHW.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 * Remark: This layer is from Torch and wrapped in Keras style.
 */
class LRN2D[T: ClassTag](
    val alpha: Double = 1e-4,
    val k: Double = 1.0,
    val beta: Double = 0.75,
    val n: Int = 5,
    val dimOrdering: DataFormat = DataFormat.NCHW,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))
    with IdentityOutputShape {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = SpatialCrossMapLRN(n, alpha, beta, k, dimOrdering)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object LRN2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    alpha: Double = 1e-4,
    k: Double = 1.0,
    beta: Double = 0.75,
    n: Int = 5,
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): LRN2D[T] = {
    new LRN2D[T](alpha, k, beta, n, KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}

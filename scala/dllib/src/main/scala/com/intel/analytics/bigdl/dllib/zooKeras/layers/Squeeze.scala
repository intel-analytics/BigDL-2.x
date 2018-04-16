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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * Delete the singleton dimension(s).
 * The batch dimension needs to be unchanged.
 * For example, if input has size (2, 1, 3, 4, 1):
 * Squeeze(dim = 1) will give output size (2, 3, 4, 1)
 * Squeeze(dims = null) will give output size (2, 3, 4)
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param dims The dimension(s) to squeeze. 0-based index. Cannot squeeze the batch dimension.
 *             The selected dimensions must be singleton, i.e. having size 1. Default is null,
 *             and in this case all the non-batch singleton dimensions will be deleted.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 * Remark: This layer is from Torch and wrapped in Keras style.
 */
class Squeeze[T: ClassTag](
    val dims: Array[Int] = null,
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) {

  if (dims != null) {
    for (dim <- dims) {
      require(dim != 0, s"Invalid squeeze dim: $dim, cannot squeeze the batch dimension")
      require(dim > 0, s"Invalid squeeze dim: $dim, dim should be positive")
    }
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val output = inputShape.toSingle().toArray.toBuffer
    if (dims != null) {
      for (dim <- dims.sortWith(_>_)) {
        require(output(dim) == 1, s"Invalid squeeze dim: $dim which has size ${output(dim)}, " +
          s"cannot squeeze a non-singleton dimension")
        require(dim <= output.length -1,
          s"Invalid squeeze dim: $dim, dim should be within range (0, ${output.length - 1}]")
        output.remove(dim)
      }
    }
    else {
      var i = output.length -1
      while (i > 0) {
        if (output(i) == 1) output.remove(i)
        i -= 1
      }
    }
    Shape(output.toArray)
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val layer = com.intel.analytics.bigdl.nn.Squeeze(dims, batchMode = true)
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Squeeze {
  def apply[@specialized(Float, Double) T: ClassTag](
    dim: Int,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Squeeze[T] = {
    new Squeeze[T](Array(dim), inputShape)
  }

  def apply[@specialized(Float, Double) T: ClassTag](
    dims: Array[Int],
    inputShape: Shape)(implicit ev: TensorNumeric[T]): Squeeze[T] = {
    new Squeeze[T](dims, inputShape)
  }
}

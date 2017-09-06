/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.pipeline.common.nn

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SpatialWithinChannelLRN[@specialized(Float, Double) T: ClassTag]
(val size: Int = 5, val alpha: Double = 1.0, val beta: Double = 0.75)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(size % 2 == 1, "LRN only supports odd values for size")
  val lrn = Sequential[T]()
    .add(new ConcatTable[T]()
      .add(Identity[T]())
      .add(Sequential[T]()
        .add(Power[T](2))
        .add(SpatialAveragePooling[T](size, size, padW = (size - 1) / 2,
          padH = (size - 1) / 2).ceil())
        .add(Power[T](-beta, alpha, 1))))
    .add(CMulTable[T]())

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = lrn.forward(input).toTensor[T]
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    throw new NotImplementedError()
  }
}

object SpatialWithinChannelLRN {

  def apply[@specialized(Float, Double) T: ClassTag](
    size: Int = 5,
    alpha: Double = 1.0,
    beta: Double = 0.75)(implicit ev: TensorNumeric[T]): SpatialWithinChannelLRN[T] = {
    new SpatialWithinChannelLRN[T](size, alpha, beta)
  }
}

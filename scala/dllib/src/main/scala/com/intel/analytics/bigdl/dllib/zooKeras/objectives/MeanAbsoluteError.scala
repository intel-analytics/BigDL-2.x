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

package com.intel.analytics.zoo.pipeline.api.keras.objectives

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.AbsCriterion

import scala.reflect.ClassTag

/**
 * A loss that measures the mean absolute value of the element-wise difference
 * between the input and the target.
 *
 * @param sizeAverage Boolean. Whether losses are averaged over observations for each
 *                    mini-batch. Default is true. If false, the losses are instead
 *                    summed for each mini-batch.
 */
class MeanAbsoluteError[@specialized(Float, Double) T: ClassTag](
      sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T])
  extends TensorLossFunction[T] {

  override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
    AbsCriterion[T](sizeAverage)
}

object MeanAbsoluteError {
  def apply[@specialized(Float, Double) T: ClassTag](
      sizeAverage: Boolean = true)
      (implicit ev: TensorNumeric[T]): MeanAbsoluteError[T] = {
    new MeanAbsoluteError[T](sizeAverage)
  }
}

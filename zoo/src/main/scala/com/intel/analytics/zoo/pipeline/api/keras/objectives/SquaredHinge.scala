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
import com.intel.analytics.bigdl.nn.MarginCriterion
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, TensorCriterion}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc4, TensorFunc6}

import scala.reflect.ClassTag
/**
 * Creates a criterion that optimizes a two-class classification (squared)
 * hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.
 *
 * When margin = 1, sizeAverage = True and squared = False, this is the same as hinge loss in keras;
 * When margin = 1, sizeAverage = False and squared = True, this is the same as squared_hinge loss
 * in keras.
 *
 * @param margin if unspecified, is by default 1.
 * @param sizeAverage whether to average the loss
 * @param squared whether to calculate the squared hinge loss
 */
class SquaredHinge[@specialized(Float, Double) T: ClassTag]
  (val margin: Double = 1.0, val sizeAverage: Boolean = true, squared: Boolean = false)
   (implicit ev: TensorNumeric[T]) extends TensorLossFunction[T] {

 override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
   MarginCriterion(margin, sizeAverage, squared)

}


object SquaredHinge {
  def apply[@specialized(Float, Double) T: ClassTag](
      margin: Double = 1.0,
      sizeAverage: Boolean = true,
      squared: Boolean = false)(implicit ev: TensorNumeric[T]) : SquaredHinge[T] = {
    new SquaredHinge[T](margin, sizeAverage, squared)
  }
}

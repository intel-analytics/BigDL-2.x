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

import com.intel.analytics.bigdl.nn.MeanAbsolutePercentageCriterion
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, TensorCriterion}
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * It caculates diff = K.abs((y - x) / K.clip(K.abs(y), K.epsilon(), Double.MaxValue))
 * and return 100 * K.mean(diff) as outpout
 */
class MeanAbsolutePercentageError[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T])
  extends TensorLossFunction[T] {

  override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
    MeanAbsolutePercentageCriterion()
}

object MeanAbsolutePercentageError {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]): MeanAbsolutePercentageError[T] = {
    new MeanAbsolutePercentageError[T]()
  }
}

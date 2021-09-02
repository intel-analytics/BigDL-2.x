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

import com.intel.analytics.bigdl.nn. PoissonCriterion
import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag
/**
 * Loss calculated as:
 * K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)
 */
class Poisson[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends TensorLossFunction[T]{

  override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
    PoissonCriterion[T]()
}

object Poisson {
  def apply[@specialized(Float, Double) T: ClassTag]()
   (implicit ev: TensorNumeric[T]): Poisson[T] = {
    new Poisson[T]()
  }
}


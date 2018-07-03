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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

abstract class NetCriterion[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
(implicit ev: TensorNumeric[T]) extends AbstractCriterion[A, B, T] {

  val loss: AbstractCriterion[A, B, T]

  override def updateOutput(input: A, target: B): T = {
    loss.updateOutput(input, target)
    output = loss.output
    output
  }

  def updateGradInput(input: A, target: B): A = {
    loss.updateGradInput(input, target)
    gradInput = loss.gradInput
    gradInput
  }

}

abstract class NetTensorCriterion[T: ClassTag]
(implicit ev: TensorNumeric[T]) extends NetCriterion[Tensor[T], Tensor[T], T]
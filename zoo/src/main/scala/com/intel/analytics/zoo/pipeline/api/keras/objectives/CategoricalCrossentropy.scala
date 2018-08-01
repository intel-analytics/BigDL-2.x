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

import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This is same with cross entropy criterion, except the target tensor is a one-hot tensor
 *
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */

class CategoricalCrossentropy[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends TensorLossFunction[T] {
  override val loss: AbstractCriterion[Tensor[T], Tensor[T], T] =
    CrossEntropyCriterion[T]()

  import CategoricalCrossentropy._

  private val buffer = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    buffer.resizeAs(input)
    output = loss.forward(buffer.log(input), convertTensor(target))
    output
  }

  override def backward(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput = loss.backward(buffer, convertTensor(target))
    gradInput.div(input)
    gradInput
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput = loss.updateGradInput(buffer, convertTensor(target))
    gradInput.div(input)
    gradInput
  }
}

object CategoricalCrossentropy {
  def apply[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]): CategoricalCrossentropy[T] = {
    new CategoricalCrossentropy[T]()
  }

  private def convertTensor[T](tensor: Tensor[T]): Tensor[T] = {
    tensor.max(2)._2
  }
}

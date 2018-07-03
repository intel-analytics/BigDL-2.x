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

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion => BigDLClassNLLCriterion}

import scala.reflect.ClassTag

class ClassNLLCriterion[T: ClassTag](
    val weights: Tensor[T] = null,
    val sizeAverage: Boolean = true,
    val logProbAsInput: Boolean = true,
    val paddingValue: Int = -1,
    val zeroBasedLabel: Boolean = true)(implicit ev: TensorNumeric[T])
  extends NetTensorCriterion[T] {

  override val loss: TensorCriterion[T] =
    BigDLClassNLLCriterion[T](weights, sizeAverage, logProbAsInput, paddingValue)

  private val buffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    if (zeroBasedLabel) {
      buffer.resizeAs(target)
      buffer.fill(ev.one).add(target)
      output = loss.updateOutput(input, buffer)
      output
    }
    else {
      output = loss.updateOutput(input, target)
      output
    }
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    if (zeroBasedLabel) {
      buffer.resizeAs(target)
      buffer.fill(ev.one).add(target)
      gradInput = loss.updateGradInput(input, buffer)
      gradInput
    }
    else {
      gradInput = loss.updateGradInput(input, target)
      gradInput
    }
  }
}

object ClassNLLCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      weights: Tensor[T] = null,
      sizeAverage: Boolean = true,
      logProbAsInput: Boolean = true,
      paddingValue: Int = -1,
      zeroBasedLabel: Boolean = true)
    (implicit ev: TensorNumeric[T]): ClassNLLCriterion[T] = {
    new ClassNLLCriterion[T](weights, sizeAverage, logProbAsInput,
      paddingValue, zeroBasedLabel)
  }
}
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

import com.intel.analytics.bigdl.nn.{SplitTable}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class InternalSplitTensor[T: ClassTag](val dimension: Int, num: Int, nested: Boolean)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Table, T] {

  def splitToTensor[@specialized(Float, Double) T: ClassTag]
  (tensor: Tensor[T], num: Int)(implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    tensor.split(tensor.size(dimension) / num, dimension)
  }

  def splitToNestedTable[@specialized(Float, Double) T: ClassTag]
  (tensor: Tensor[T], num: Int)(implicit ev: TensorNumeric[T]): Table = {
    val states = T.array(tensor.split(tensor.size(dimension) / num, dimension))
    var i = 1
    while (i <= states.length()) {
      val state = states[Tensor[T]](i)
      states(i) = T.array(state.split(state.size(dimension) / 2, dimension))
      i += 1
    }
    states
  }

  override def updateOutput(input: Tensor[T]): Table = {
    output = if (!nested) {
      T.array(splitToTensor(input, num))
    } else splitToNestedTable(input, num)
    output
  }

  private val innerLayer = new InternalJoinTable[T](dimension, -1)
  override def updateGradInput(input: Tensor[T], gradOutput: Table): Tensor[T] = {
    gradInput = innerLayer.forward(gradOutput).toTensor[T]
    gradInput
  }
}

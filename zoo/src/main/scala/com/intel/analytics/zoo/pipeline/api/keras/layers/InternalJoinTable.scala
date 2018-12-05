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

import com.intel.analytics.bigdl.nn.JoinTable
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class InternalJoinTable[T: ClassTag](val dimension: Int,
  val nInputDims: Int)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[_], T] {

  private val innerLayer = JoinTable[T](dimension, nInputDims)
  var flat: Table = null

  override def updateOutput(input: Table): Tensor[_] = {
    flat = input.flatten()
    output = innerLayer.forward(flat)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[_]): Table = {
    val g = innerLayer.updateGradInput(flat, gradOutput)
    gradInput = T(g.inverseFlatten(input), T())
    gradInput
  }
}

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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This module is for debug purpose, which can print activation and gradient in your model
 * topology
 */
@SerialVersionUID(6735245897546687343L)
class Echo[T: ClassTag] (implicit ev: TensorNumeric[T])
  extends TensorModule[T]  {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    this.output = input
    println(s"${getPrintName} : Activation size is ${input.size().mkString("x")}")
    this.output
  }
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput = gradOutput
    println(s"${getPrintName} : Gradient size is ${gradOutput.size().mkString("x")}")
    this.gradInput
  }
}

object Echo {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : Echo[T] = {
    new Echo[T]()
  }
}

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

package com.intel.analytics.bigdl.pipeline.common.nn

import com.intel.analytics.bigdl.nn.{CMul, Normalize}
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

class NormalizeScale[T: ClassTag](val p: Double, val eps: Double = 1e-10,
  val scale: Double, size: Array[Int])(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  val normalize = Normalize[T](p, eps)
  val cmul = CMul[T](size)
  cmul.weight.fill(ev.fromType(scale))

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    normalize.forward(input)
    output = cmul.forward(normalize.output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = cmul.backward(output, normalize.backward(input, gradOutput))
    gradInput
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(cmul.weight), Array(cmul.gradWeight))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> cmul.weight, "gradWeight" -> cmul.gradWeight))
  }
}

object NormalizeScale {
  def apply[@specialized(Float, Double) T: ClassTag]
  (p: Double, eps: Double = 1e-10, scale: Double, size: Array[Int])
    (implicit ev: TensorNumeric[T]): NormalizeScale[T] = new NormalizeScale[T](p, eps, scale, size)
}

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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Takes a table with two Tensor and returns the component-wise division between them.
 */

@SerialVersionUID(- 3746356029327536265L)
class CDivTable[T: ClassTag](implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Tensor[T], T]{

  override def updateOutput(input: Table): Tensor[T] = {
    val res1 = input[Tensor[T]](1)
    val res2 = input[Tensor[T]](2)

    output.resizeAs(res1).copy(res1)
    output.cdiv(res2)
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val res1 = input[Tensor[T]](1)
    val res2 = input[Tensor[T]](2)

    if (!gradInput.contains(1)) gradInput.insert(1, Tensor[T]())
    if (!gradInput.contains(2)) gradInput.insert(2, Tensor[T]())
    gradInput[Tensor[T]](1).resizeAs(res1).copy(gradOutput).cdiv(res2)
    gradInput[Tensor[T]](2).resizeAs(res2).zero().
      addcdiv(ev.fromType(-1), gradInput(1), res2).cmul(res1)

    gradInput
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[CDivTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: CDivTable[T] =>
      super.equals(that) &&
        (that canEqual this)
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode())
    state.map(getHashCode).foldLeft(0)((a, b) => 37 * a + b)
  }
}

object CDivTable {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : CDivTable[T] = {
    new CDivTable[T]()
  }
}

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

import scala.reflect.ClassTag

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

/**
 * Reverse the input w.r.t given dimension.
 * The input can be a Tensor or Table.
 * @param dim
 * @param ev
 * @tparam T Numeric type. Only support float/double now
 */
class ReverseDS[T: ClassTag](dim: Int = 1) (implicit ev: TensorNumeric[T])
    extends TensorModule[T] {

  val buffer = Tensor[T]()

  /**
   * reverse the src Tensor and write it to target w.r.t given dim.
   * E.g. src: (1,2,3; 4,5,6) and dim = 1
   *      target: (4,5,6; 1,2,3)
   * @param src
   * @param dim
   * @return
   */
  private def reverseTensor(src: Tensor[T], dim: Int): Tensor[T] = {
    require(dim > 0 && dim <= src.dim,
      s"Reverse: the designated dimension ${dim} to reverse input Tensor" +
        s" is out of index. The input.dim = ${src.dim}")

    val time = src.size(dim)
    val half = time >> 1
    buffer.resizeAs(src.select(dim, 1))
    var i = 1
    while (i <= half) {
      buffer.copy(src.select(dim, time - i + 1))
      src.select(dim, time - i + 1).copy(src.select(dim, i))
      src.select(dim, i).copy(buffer)
      i += 1
    }
    src
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = reverseTensor(input.toTensor[T], dim)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = reverseTensor(gradOutput.toTensor[T], dim)
    gradInput
  }

  override def clearState(): ReverseDS.this.type = {
    super.clearState()
    buffer.set()
    this
  }

  override def toString: String = s"nn.Reverse"

  override def equals(other: Any): Boolean = super.equals(other)

  override def hashCode(): Int = super.hashCode()

  override def canEqual(other: Any): Boolean = other.isInstanceOf[JoinTable[T]]

}

object ReverseDS {
  def apply[@specialized(Float, Double) T: ClassTag](
      dimension: Int = 1)(implicit ev: TensorNumeric[T]) : ReverseDS[T] = {
    new ReverseDS[T](dimension)
  }
}

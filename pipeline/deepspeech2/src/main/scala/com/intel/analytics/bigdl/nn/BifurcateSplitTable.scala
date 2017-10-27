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
 * Creates a module that takes a Tensor as input and
 * outputs two tables, splitting the Tensor along
 * the specified dimension `dimension`.
 *
 * The input to this layer is expected to be a tensor, or a batch of tensors;
 *
 * @param dimension to be split along this dimension
 * @tparam T Numeric type. Only support float/double now
 */

@SerialVersionUID( 6506746847756687944L)
class BifurcateSplitTable[T: ClassTag](
  var dimension: Int)
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Table, T]{

  override def updateOutput(input: Tensor[T]): Table = {
    val slices = input.size(dimension)
    require(slices >= 1,
      s"BifurcateSplitTable: the size of referred dimension is ${slices}. " +
        s"It should be larger than 1.")
    val leftSlices = slices >> 1
    val rightSlices = slices - leftSlices

    output(1) = input.narrow(dimension, 1, leftSlices).contiguous()
    output(2) = input.narrow(dimension, 1 + leftSlices, rightSlices).contiguous()
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Table): Tensor[T] = {
    val slices = input.size(dimension)
    val leftSlices = slices >> 1
    val rightSlices = slices - leftSlices

    gradInput.resizeAs(input)

    gradInput.narrow(dimension, 1, leftSlices).copy(gradOutput(1))
    gradInput.narrow(dimension, 1 + leftSlices, rightSlices).copy(gradOutput(2))

    gradInput
  }


  override def canEqual(other: Any): Boolean = other.isInstanceOf[SplitTable[T]]


  override def toString: String = s"BifurcateSplitTable($dimension)"

  override def equals(other: Any): Boolean = other match {
    case that: BifurcateSplitTable[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        dimension == that.dimension
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), dimension)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object BifurcateSplitTable {
  def apply[@specialized(Float, Double) T: ClassTag](
    dimension: Int)(implicit ev: TensorNumeric[T]) : BifurcateSplitTable[T] = {
    new BifurcateSplitTable[T](dimension)
  }
}

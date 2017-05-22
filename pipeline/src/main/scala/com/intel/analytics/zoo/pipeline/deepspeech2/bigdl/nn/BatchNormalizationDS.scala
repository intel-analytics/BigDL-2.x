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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

@SerialVersionUID( - 467695939363389565L)
class BatchNormalizationDS[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-3,
    momentum: Double = 0.1,
    affine: Boolean = false)
    (implicit ev: TensorNumeric[T]
    ) extends BatchNormalization[T](nOutput, eps, momentum, affine) {

  val batchDim = 0
  val timeDim = 1
  val featDim = 2

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 3,
      "In BatchNormalizationBRNN, input should be a 3D tensor, [B, T, D]" +
        s"input.dim = ${input.dim}")
    val size = input.size()
    input.resize(size(batchDim) * size(timeDim), size(featDim))
    output = super.updateOutput(input)
    output.resize(size)
    input.resize(size)
    output
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val size = input.size()
    input.resize(size(batchDim) * size(timeDim), size(featDim))
    gradOutput.resize(size(batchDim) * size(timeDim), size(featDim))
    val result = super.backward(input, gradOutput)
    input.resize(size)
    gradOutput.resize(size)
    result.resize(size)
    result
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val size = input.size()
    input.resize(size(batchDim) * size(timeDim), size(featDim))
    gradOutput.resize(size(batchDim) * size(timeDim), size(featDim))
    gradInput = super.updateGradInput(input, gradOutput)
    gradInput.resize(size)
    input.resize(size)
    gradOutput.resize(size)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double): Unit = {
    val size = input.size()
    input.resize(size(batchDim) * size(timeDim), size(featDim))
    gradOutput.resize(size(batchDim) * size(timeDim), size(featDim))
    super.accGradParameters(input, gradOutput)
    input.resize(size)
    gradOutput.resize(size)
  }

  override def toString(): String = s"BatchNormalizationBRNN"

  override def canEqual(other: Any): Boolean = other.isInstanceOf[BatchNormalizationDS[T]]

  override def equals(other: Any): Boolean = other match {
    case that: BatchNormalizationDS[T] =>
      super.equals(that) &&
        (that canEqual this)
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), batchDim, timeDim, featDim)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object BatchNormalizationDS {
  def apply[@specialized(Float, Double) T: ClassTag](
      nOutput: Int,
      eps: Double = 1e-5) (implicit ev: TensorNumeric[T]): BatchNormalizationDS[T] = {
    new BatchNormalizationDS[T](nOutput, eps)
  }
}

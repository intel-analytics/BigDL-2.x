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

import com.intel.analytics.bigdl.nn.{Cell, ConvLSTMPeephole3D}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

// only support channel_first: (batch, time, channels, dim1, dim2, dim3)
// currently only support same padding. TODO: test for valid padding
class ConvLSTM3D[T: ClassTag](
    val nbFilter: Int,
    val nbKernel: Int, // both square kernels
    val subsample: Int = 1, // same stride for all dimensions
    var wRegularizer: Regularizer[T] = null,
    var uRegularizer: Regularizer[T] = null,
    var bRegularizer: Regularizer[T] = null,
    override val returnSequences: Boolean = false,
    override val goBackwards: Boolean = false,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T]) extends Recurrent[T] (
  nbFilter, returnSequences, goBackwards, inputShape) with Net {

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 6,
      s"ConvLSTM3D requires 6D input, but got input dim ${input.length}")
    val outDim1 = KerasUtils.computeConvOutputLength(input(3), nbKernel, "same", subsample)
    val outDim2 = KerasUtils.computeConvOutputLength(input(4), nbKernel, "same", subsample)
    val outDim3 = KerasUtils.computeConvOutputLength(input(5), nbKernel, "same", subsample)
    if (returnSequences) Shape(input(0), input(1), nbFilter, outDim1, outDim2, outDim3)
    else Shape(input(0), nbFilter, outDim1, outDim2, outDim3)
  }

  override def buildCell(input: Array[Int]): Cell[T] = {
    ConvLSTMPeephole3D(
      inputSize = input(2),
      outputSize = nbFilter,
      kernelI = nbKernel,
      kernelC = nbKernel,
      stride = subsample,
      wRegularizer = wRegularizer,
      uRegularizer = uRegularizer,
      bRegularizer = bRegularizer,
      withPeephole = false)
  }
}

object ConvLSTM3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbKernel: Int,
    subsample: Int = 1,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    returnSequences: Boolean = false,
    goBackwards: Boolean = false,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ConvLSTM3D[T] = {
    new ConvLSTM3D[T](nbFilter, nbKernel, subsample, wRegularizer,
      uRegularizer, bRegularizer, returnSequences, goBackwards, inputShape)
  }
}
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
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, Tensor, TensorFunc6}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This class is a transform layer corresponding to the sigmoid function:
 *  f(x) = Log(1 / (1 + e ^^ (-x)))
 */

@SerialVersionUID(884823114984663135L)
class LogSigmoid[T: ClassTag] (implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  @transient private var buffer: Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (buffer == null) {
      buffer = Tensor[T]()
    }

    output.resizeAs(input)
    buffer.resizeAs(input)

    // Todo: Replace apply to get a better performance
    val func = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
        data3: Array[T], offset3: Int): Unit = {
        val z = ev.exp(ev.negative(data2(offset2)))
        data3(offset3) = z
        data1(offset1) = ev.negative(ev.log1p(z))
      }
    }
    DenseTensorApply.apply3[T](output, input, buffer, func)

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.isSameSizeAs(gradOutput), "input and gradOutput should have the same size")
    gradInput
      .resizeAs(buffer)

    // Todo: Replace apply to get a better performance
    val func = new TensorFunc6[T] {
      override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int,
        data3: Array[T], offset3: Int): Unit = {
        val z = data3(offset3)
        data1(offset1) = ev.divide(
          ev.times(data2(offset2), z), ev.plus(ev.fromType[Int](1), z))
      }
    }
    DenseTensorApply.apply3[T](gradInput, gradOutput, buffer, func)

    gradInput
  }
}

object LogSigmoid {
  def apply[@specialized(Float, Double) T: ClassTag]()
      (implicit ev: TensorNumeric[T]) : LogSigmoid[T] = {
    new LogSigmoid[T]()
  }
}

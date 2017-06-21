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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Extracts a strided slice from a tensor.
 * @param sliceSpecs Array(dim, begin_index, end_index, stride)
 */
@SerialVersionUID(4436600172725317184L)
private[bigdl] class StrideSlice[T: ClassTag](sliceSpecs: Array[(Int, Int, Int, Int)])
                (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(sliceSpecs.map(_._4 == 1).reduce(_ && _), "only support stride 1 for now")

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    var tmp = input
    var i = 0
    while(i < sliceSpecs.length) {
      tmp = tmp.narrow(sliceSpecs(i)._1, sliceSpecs(i)._2, sliceSpecs(i)._3 - sliceSpecs(i)._2)
      i += 1
    }
    output.resizeAs(tmp)
    output.copy(tmp)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    gradInput.zero()
    var tmp = gradInput
    var i = 0
    while(i < sliceSpecs.length) {
      tmp = tmp.narrow(sliceSpecs(i)._1, sliceSpecs(i)._2, sliceSpecs(i)._3 - sliceSpecs(i)._2)
      i += 1
    }
    tmp.copy(gradOutput)
    gradInput
  }

}

private[bigdl] object StrideSlice {
  def apply[T: ClassTag](sliceSpecs: Array[(Int, Int, Int, Int)])
       (implicit ev: TensorNumeric[T]) : StrideSlice[T] = {
    new StrideSlice[T](sliceSpecs)
  }
}

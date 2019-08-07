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

package com.intel.analytics.bigdl.tensor

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

object BigDLTensorExtension {

  def subTensor[T: ClassTag](tensor: Tensor[T], tensor2: Tensor[T])
                            (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val expandTensor = tensor.asInstanceOf[DenseTensor[T]]
      .expandTensor(tensor2.asInstanceOf[DenseTensor[T]]).contiguous()
    tensor.sub(expandTensor)
    tensor
  }

  def divTensor[T: ClassTag](tensor: Tensor[T], tensor2: Tensor[T])
                            (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val expandTensor = tensor.asInstanceOf[DenseTensor[T]]
      .expandTensor(tensor2.asInstanceOf[DenseTensor[T]]).contiguous()
    tensor.div(expandTensor)
    tensor
  }
}

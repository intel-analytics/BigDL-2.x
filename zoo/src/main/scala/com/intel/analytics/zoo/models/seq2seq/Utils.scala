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

package com.intel.analytics.zoo.models.seq2seq

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

private[seq2seq] object Utils {
  def join[@specialized(Float, Double) T: ClassTag]
  (tensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val sizes = tensors.head.size()
    val newTensor = Tensor[T](Array(sizes(0), tensors.size * sizes(1)) ++ sizes.drop(2))

    for (i <- 0 until tensors.size) {
      newTensor.narrow(2, i * sizes(1) + 1, sizes(1)).copy(tensors(i))
    }
    newTensor
  }

  def join[@specialized(Float, Double) T: ClassTag]
  (tables: Array[Table])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val size = tables.head.length()
    require(size == 2, "only support cat table with size 2")
    val tensor1 = join(tables.map(_[Tensor[T]](1)))
    val tensor2 = join(tables.map(_[Tensor[T]](2)))

    join(Array(tensor1, tensor2))
  }

  def join[@specialized(Float, Double) T: ClassTag]
  (activities: Array[Activity])(implicit ev: TensorNumeric[T]): Activity = {
    if (activities.head.isTensor) join(activities.map(_.toTensor))
    else join(activities.map(_.toTable))
  }

  def splitToTensor[@specialized(Float, Double) T: ClassTag]
  (tensor: Tensor[T], num: Int)(implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    tensor.split(tensor.size(2) / num, 2)
  }

  def splitToTable[@specialized(Float, Double) T: ClassTag]
  (tensor: Tensor[T], num: Int)(implicit ev: TensorNumeric[T]): Array[Table] = {
    val states = splitToTensor(tensor, 2)

    val data1 = states(0).split(states(0).size(2) / num, 2)
    val data2 = states(1).split(states(1).size(2) / num, 2)
    data1.zip(data2).map(x => T(x._1, x._2))
  }
}

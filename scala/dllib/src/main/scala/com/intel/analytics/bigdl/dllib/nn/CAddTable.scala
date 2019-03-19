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
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable}

import scala.reflect._

/**
 * Merge the input tensors in the input table by element wise adding them together. The input table
 * is actually an array of tensor with same size.
 * @param inplace reuse the input memory
 * @param ev numeric operator
 * @tparam T Numeric type. Only support float/double now
 */
@SerialVersionUID(7959261460060075605L)
class CAddTable[T: ClassTag, D: ClassTag](val inplace: Boolean = false)(
  implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends AbstractModule[Table, Tensor[D], T] with MklInt8Convertible {

  output = Tensor[D]()

  override def updateOutput(input: Table): Tensor[D] = {
    var scalar = ev2.zero
    var hasTensor = false
    var hasScalar = false
    var initTensor = false

    var i = 1
    while (i <= input.length()) {
      val curTensor = input[Tensor[D]](i)
      if (curTensor.isScalar) {
        scalar = ev2.plus(scalar, curTensor.value())
        hasScalar = true
      } else if (curTensor.isTensor) {
        if (initTensor) {
          output = output.add(curTensor)
        } else {
          if (inplace) {
            output.set(curTensor)
          } else {
            output.resizeAs(curTensor).copy(curTensor)
          }
          initTensor = true
        }
        hasTensor = true
      }
      i += 1
    }

    if (hasTensor && hasScalar) {
      output.add(scalar)
    } else if (hasScalar) {
      if (inplace) {
        output.set(input[Tensor[D]](1)).setValue(scalar)
      } else {
        output.resizeAs(input[Tensor[D]](1)).setValue(scalar)
      }
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[D]) : Table = {
    var i = 1
    var sum = ev2.zero
    var calculateSum = false
    while (i <= input.length()) {
      if (i > gradInput.length) gradInput.insert(i, Tensor[T]().resizeAs(input(1)))
      if (inplace) {
        require(input[Tensor[D]](1).isSameSizeAs(gradOutput), "cannot use inplace for broadcast")
        gradInput[Tensor[D]](i).set(gradOutput)
      } else {
        if (input[Tensor[D]](i).isSameSizeAs(gradOutput)) {
          gradInput[Tensor[D]](i).resizeAs(gradOutput).copy(gradOutput)
        } else {
          require(input[Tensor[D]](i).isScalar, "Only support scalar broadcast backward now")
          if (!calculateSum) {
            sum = gradOutput.sum()
            calculateSum = true
          }
          gradInput[Tensor[D]](i).resizeAs(input[Tensor[D]](i)).setValue(sum)
        }
      }
      i += 1
    }
    i = input.length + 1
    while (i <= gradInput.length) {
      gradInput.remove(i)
    }
    gradInput
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    this
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}


object CAddTable extends ModuleSerializable {
  def apply[T: ClassTag](
      inplace: Boolean = false)(implicit ev: TensorNumeric[T]) : CAddTable[T, T] = {
    new CAddTable[T, T](inplace)
  }

  override def getTypes(context: DeserializeContext): (Array[ClassTag[_]],
    Array[TensorNumeric[_]]) = {
    var (tags, numerics) = super.getTypes(context)
    val defaultTag = tags(0)
    val defaultNumeric = numerics(0)
    if (tags.size < 2) {
      val extendedTags = Array[ClassTag[_]](defaultTag, defaultTag)
      val extendNumerics = Array[TensorNumeric[_]](defaultNumeric, defaultNumeric)
      (extendedTags, extendNumerics)
    } else {
      (tags, numerics)
    }
  }
}



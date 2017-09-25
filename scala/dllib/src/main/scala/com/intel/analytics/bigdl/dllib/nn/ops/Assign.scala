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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Update 'ref' by assigning 'value' to it.
 *
 * This operation outputs a Tensor that holds the new value of 'ref' after
 * the value has been assigned.
 * This makes it easier to chain operations that need to use the reset value.
 *
 * The `input` has two elements, the first one is `ref`, the second is `value`.
 *
 * @param validateShape An optional bool. Defaults to True.
 *                      If true, the operation will validate that the shape of
 *                      'value' matches the shape of the Tensor being assigned to.
 *                      If false, 'ref' will take on the shape of 'value'.
 * @param useLocking An optional bool. Defaults to True.
 *                   If True, the assignment will be protected by a lock;
 *                   otherwise the behavior is undefined, but may exhibit less contention.
 *
 * @tparam T Numeric type. Only support float/double now
 */
class Assign[T: ClassTag](
  validateShape: Boolean = true,
  useLocking: Boolean = true
)
  (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[_], T] {

  override def updateOutput(input: Table): Tensor[_] = {
    val input1 = input[Tensor[NumericWildcard]](1)
    val input2 = input[Tensor[NumericWildcard]](2)

    require(input1.getType() == input2.getType(),
      "ref and value must have the same tensor numeric type")

    if (validateShape) {
      var i = 1
      while (i <= input1.dim()) {
        require(input1.size(i) == input2.size(i), "shape of the ref and value are not same")
        i += 1
      }
    }

    input1
      .resizeAs(input2)
      .copy(input2)

    output.asInstanceOf[Tensor[NumericWildcard]]
      .resizeAs(input2)
      .copy(input2)
  }
}

object Assign {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Operation[Activity, Activity, T]
  = ModuleToOperation[T](new Assign())
}

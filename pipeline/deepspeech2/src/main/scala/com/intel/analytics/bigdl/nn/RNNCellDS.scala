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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.optim.Regularizer

import scala.reflect.ClassTag

@SerialVersionUID( 5237686508074490666L)
class RnnCellDS[T : ClassTag] (
   inputSize: Int = 4,
   hiddenSize: Int = 3,
   activation: TensorModule[T],
   uRegularizer: Regularizer[T] = null)
 (implicit ev: TensorNumeric[T])
  extends RnnCell[T](inputSize, hiddenSize, activation, uRegularizer = uRegularizer) {

  // preTopology should be null for DeepSpeech model
  preTopology = null

  override def toString(): String = {
    val str = "nn.RnnCellDS"
    str
  }
}

object RnnCellDS {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3,
    activation: TensorModule[T],
    uRegularizer: Regularizer[T] = null)
  (implicit ev: TensorNumeric[T]) : RnnCellDS[T] = {
    new RnnCellDS[T](inputSize, hiddenSize, activation, uRegularizer)
  }
}

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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.nn.{Input => TInput}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, Shape}

import scala.reflect.ClassTag

class Input[T: ClassTag](val inputShape: Shape)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](KerasLayer.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = inputShape

  override def doBuild(inputShape: Shape): TInput[T] = new TInput[T]()

  override def allowRebuilt(): Boolean = true
}

object Input {
  def apply[T: ClassTag](
    name : String = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    val module = new Input(inputShape)
    module.build(KerasLayer.addBatch(inputShape))
    if (name != null) {
      module.setName(name)
    }
    new Node(module.asInstanceOf[AbstractModule[Activity, Activity, T]])
  }
}

object InputLayer {
  def apply[T: ClassTag](
    name : String = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): KerasLayer[Activity, Activity, T] = {
    val module = new Input(inputShape)
    if (name != null) {
      module.setName(name)
    }
    module
  }
}

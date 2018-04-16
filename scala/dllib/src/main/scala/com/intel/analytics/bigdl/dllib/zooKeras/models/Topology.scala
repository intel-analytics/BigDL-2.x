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

package com.intel.analytics.zoo.pipeline.api.keras.models

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.keras.{Model => BModel, Sequential => BSequential}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Container for a sequential model.
 */
class Sequential[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends BSequential[T] {
}

object Sequential {
  def apply[@specialized(Float, Double) T: ClassTag]()
  (implicit ev: TensorNumeric[T]): Sequential[T] = {
    new Sequential[T]()
  }
}

/**
 * Container for a graph model.
 *
 * @param _inputs An input node or an array of input nodes.
 * @param _outputs An output node or an array of output nodes.
 */
class Model[T: ClassTag](
    private val _inputs: Seq[ModuleNode[T]],
    private val _outputs: Seq[ModuleNode[T]])(implicit ev: TensorNumeric[T])
  extends BModel[T](_inputs, _outputs) {
}

object Model {
  def apply[T: ClassTag](input: Array[ModuleNode[T]], output: Array[ModuleNode[T]])
    (implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](input, output)
  }

  def apply[T: ClassTag](input: ModuleNode[T], output: Array[ModuleNode[T]])
    (implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](Seq(input), output)
  }

  def apply[T: ClassTag](input: Array[ModuleNode[T]], output: ModuleNode[T])
    (implicit ev: TensorNumeric[T]): Model[T] = {
    new Model[T](input, Seq(output))
  }

  def apply[T: ClassTag](input: ModuleNode[T], output: ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](Seq(input), Seq(output))
  }

}

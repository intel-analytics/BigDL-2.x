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
package com.intel.analytics.zoo.pipeline.api.keras2.layers

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.Merge

import scala.reflect.ClassTag

/**
 * Layer that computes the average (element-wise) a list of inputs.
 *
 * It takes as input a list of nodes,
 * all of the same shape, and returns
 * a single node (also of the same shape).
 */
class Average[T: ClassTag](override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Merge[T](layers = null, mode = "ave", inputShape = inputShape)
    with Net {
}
object Average{
  def apply[@specialized(Float, Double) T: ClassTag]
  (inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Average[T] = {
    new Average[T](inputShape)
  }
  def average[@specialized(Float, Double) T: ClassTag](inputs: List[ModuleNode[T]])
  (implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    val layer = new Average[T]()
    layer.inputs(inputs.toArray)
  }
}

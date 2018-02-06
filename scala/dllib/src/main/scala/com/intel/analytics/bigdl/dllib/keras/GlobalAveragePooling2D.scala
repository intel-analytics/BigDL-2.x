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

import com.intel.analytics.bigdl.nn.{SpatialAveragePooling, Squeeze, Sequential => TSequential}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class GlobalAveragePooling2D[T: ClassTag](
   format: DataFormat = DataFormat.NCHW,
   inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends GlobalPooling2D[T](format, inputShape) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val (dimH, dimW, dimC) = format.getHWCDims(4)
    val model = TSequential[T]()
    val layer = SpatialAveragePooling(
      kW = input(dimW -1),
      kH = input(dimH -1),
      dW = input(dimW -1),
      dH = input(dimH -1),
      countIncludePad = false,
      format = format)
    model.add(layer)
    model.add(Squeeze(dimW))
    model.add(Squeeze(dimH))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object GlobalAveragePooling2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    dimOrdering: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): GlobalAveragePooling2D[T] = {
    new GlobalAveragePooling2D[T](KerasUtils.toBigDLFormat(dimOrdering), inputShape)
  }
}

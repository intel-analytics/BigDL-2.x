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

package com.intel.analytics.zoo.pipeline.common.caffe

import com.google.protobuf.GeneratedMessage
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.RoiPooling
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import pipeline.caffe.Caffe.LayerParameter

import scala.reflect.ClassTag


class RoiPoolingConverter[T: ClassTag](implicit ev: TensorNumeric[T]) extends Customizable[T] {
  override def convertor(layer: GeneratedMessage): Seq[ModuleNode[T]] = {
    val param = layer.asInstanceOf[LayerParameter].getRoiPoolingParam
    val poolH = param.getPooledH
    val poolW = param.getPooledW
    val scale = param.getSpatialScale
    Seq(RoiPooling[T](poolW, poolH, ev.fromType(scale)).setName(getLayerName(layer)).inputs())
  }

}

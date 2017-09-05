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
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.zoo.pipeline.common.nn.PriorBox
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import pipeline.ssd.caffe.Caffe.{LayerParameter, NetParameter}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

class PriorBoxConvertor[T: ClassTag](implicit ev: TensorNumeric[T]) extends Customizable[T] {
  override def convertor(layer: GeneratedMessage): Seq[ModuleNode[T]] = {
    val netparam = contexts("netparam").asInstanceOf[NetParameter]
    val (imgH, imgW) = if (netparam.getInputShapeCount > 0) {
      (netparam.getInputShape(0).getDim(2).toInt, netparam.getInputShape(0).getDim(3).toInt)
    } else {
      val data = contexts("name2LayerV2").asInstanceOf[Map[String, Any]]("data")
        .asInstanceOf[LayerParameter].getTransformParam.getResizeParam
      (data.getHeight, data.getWidth)
    }
    val param = layer.asInstanceOf[LayerParameter].getPriorBoxParam
    val minSizes = param.getMinSizeList.asScala.toArray.map(_.toFloat)
    val maxSizes = param.getMaxSizeList.asScala.toArray.map(_.toFloat)
    val aspectRatios = param.getAspectRatioList.asScala.toArray.map(_.toFloat)
    val isFlip = param.getFlip
    val isClip = param.getClip
    val variances = param.getVarianceList.asScala.toArray.map(_.toFloat)
    val offset = param.getOffset
    val imgSize = param.getImgSize
    val stepH = param.getStepH
    val stepW = param.getStepW
    val step = param.getStep
    Seq(PriorBox[T](minSizes,
      maxSizes,
      aspectRatios, isFlip, isClip,
      variances, offset,
      imgH, imgW, imgSize,
      stepH, stepW, step).setName(getLayerName(layer)).inputs())
  }

  def getLayerName(layer: GeneratedMessage): String = {
    layer.asInstanceOf[LayerParameter].getName
  }
}

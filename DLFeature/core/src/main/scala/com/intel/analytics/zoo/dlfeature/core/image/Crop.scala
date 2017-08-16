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

package com.intel.analytics.zoo.dlfeature.core.image

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.dlfeature.core.label.LabelTransformMethod
import com.intel.analytics.zoo.dlfeature.core.util.{BboxUtil, MatWrapper, NormalizedBox}
import org.opencv.core.Rect

class Crop(useNormalized: Boolean = true,
  bbox: Option[NormalizedBox] = None, roiKey: Option[String] = None,
  roiGenerator: Option[(Feature => NormalizedBox)] = None,
  labelTransformMethod: Option[LabelTransformMethod] = None)
  extends ImageTransformer {

  if (labelTransformMethod.isDefined) {
    setLabelTransfomer(labelTransformMethod.get)
  }

  override def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean = {
    val cropBox = if (bbox.isDefined) {
      bbox.get
    } else if (roiKey.isDefined) {
      var roi = feature(roiKey.get).asInstanceOf[Tensor[Float]]
      if (roi.dim() == 1) {
        roi = BboxUtil.decodeRois(roi)
      }
      if (roi.nElement() > 0) {
        NormalizedBox(roi.valueAt(1, 3), roi.valueAt(1, 4), roi.valueAt(1, 5), roi.valueAt(1, 6))
      } else {
        NormalizedBox(0, 0, 1, 1)
      }
    } else if (roiGenerator.isDefined) {
      require(feature.hasLabel())
      roiGenerator.get(feature)
    } else {
      throw new Exception("not supported")
    }
    Crop.transform(input, output, cropBox, useNormalized)
    feature(Feature.height) = output.height()
    feature(Feature.width) = output.width()
    if (feature.hasLabel()) {
      feature("bbox") = cropBox
    }
    true
  }
}

object Crop {
  def apply(useNormalized: Boolean = true,
    bbox: Option[NormalizedBox] = None, roiKey: Option[String] = None,
    roiGenerator: Option[(Feature => NormalizedBox)] = None,
    labelTransformMethod: Option[LabelTransformMethod] = None): Crop
  = new Crop(useNormalized, bbox, roiKey, roiGenerator, labelTransformMethod)

  def transform(input: MatWrapper, output: MatWrapper, bbox: NormalizedBox,
    useNormalized: Boolean = true): Boolean = {
    val crop = if (useNormalized) {
      val width = input.width
      val height = input.height
      bbox.clipBox(bbox)
      val scaledBox = new NormalizedBox()
      bbox.scaleBox(height, width, scaledBox)
      scaledBox
    } else {
      bbox
    }
    val rect = new Rect(crop.x1.toInt, crop.y1.toInt,
      crop.width().toInt, crop.height().toInt)
    input.submat(rect).copyTo(output)
    true
  }
}

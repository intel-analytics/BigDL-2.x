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

package com.intel.analytics.zoo.models.objectdetection.utils

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}

case class ScaleDetection() extends FeatureTransformer {
  override def transform(imageFeature: ImageFeature): ImageFeature = {
    val detection = imageFeature[Tensor[Float]](ImageFeature.predict)
    // Scale the bbox according to the original image size.
    val height = imageFeature.getOriginalHeight
    val width = imageFeature.getOriginalWidth
    val result = BboxUtil.decodeRois(detection)
    if (result.nElement() > 0) {
      // clipBoxes to [0, 1]
      clipBoxes(result.narrow(2, 3, 4))
      // scaleBoxes
      result.select(2, 3).mul(width)
      result.select(2, 4).mul(height)
      result.select(2, 5).mul(width)
      result.select(2, 6).mul(height)
    }
    imageFeature(ImageFeature.predict) = result
    imageFeature
  }

  def clipBoxes(bboxes: Tensor[Float]): Tensor[Float] = {
    bboxes.cmax(0).apply1(x => Math.min(1, x))
  }
}

case class DecodeOutput() extends FeatureTransformer {
  override def transform(imageFeature: ImageFeature): ImageFeature = {
    val detection = imageFeature[Tensor[Float]](ImageFeature.predict)
    val result = BboxUtil.decodeRois(detection)
    imageFeature(ImageFeature.predict) = result
    imageFeature
  }
}


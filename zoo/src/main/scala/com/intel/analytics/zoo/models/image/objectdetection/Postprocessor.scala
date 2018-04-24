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

package com.intel.analytics.zoo.models.objectdetection

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}

/**
 * If the detection is normalized, for example, ssd detected bounding box is in [0, 1],
 * need to scale the bbox according to the original image size.
 * Note that in this transformer, the tensor from model output will be decoded,
 * just like `DecodeOutput`
 */
case class ScaleDetection() extends FeatureTransformer {
  override def transformMat(imageFeature: ImageFeature): Unit = {
    val detection = imageFeature[Tensor[Float]](ImageFeature.predict)
    // Scale the bbox according to the original image size.
    val height = imageFeature.getOriginalHeight
    val width = imageFeature.getOriginalWidth
    val result = BboxUtil.decodeRois(detection)
    if (result.dim() == 2 && result.nElement() > 0) {
      // clipBoxes to [0, 1]
      clipBoxes(result.narrow(2, 3, 4))
      // scaleBoxes
      result.select(2, 3).mul(width)
      result.select(2, 4).mul(height)
      result.select(2, 5).mul(width)
      result.select(2, 6).mul(height)
    }
    imageFeature(ImageFeature.predict) = result
  }

  def clipBoxes(bboxes: Tensor[Float]): Tensor[Float] = {
    bboxes.cmax(0).apply1(x => Math.min(1, x))
  }
}

/**
 * Decode the detection output
 * The output of the model prediction is a 1-dim tensor
 * The first element of tensor is the number(K) of objects detected,
 * followed by [label score x1 y1 x2 y2] * K
 * For example, if there are 2 detected objects, then K = 2, the tensor may
 * looks like
 * ```2, 1, 0.5, 10, 20, 50, 80, 3, 0.3, 20, 10, 40, 70```
 * After decoding, it returns a 2-dim tensor, each row represents a detected object
 * ```
 * 1, 0.5, 10, 20, 50, 80
 * 3, 0.3, 20, 10, 40, 70
 * ```
 */
case class DecodeOutput() extends FeatureTransformer {
  override def transformMat(imageFeature: ImageFeature): Unit = {
    val detection = imageFeature[Tensor[Float]](ImageFeature.predict)
    val result = BboxUtil.decodeRois(detection)
    imageFeature(ImageFeature.predict) = result
  }
}


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

package com.intel.analytics.zoo.pipeline.utils

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.util.BoundingBox
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}

class ImageMeta(numClass: Int) extends FeatureTransformer {
  override def transformMat(feature: ImageFeature): Unit = {
    val meta = Tensor[Float](8 + numClass)
    meta.setValue(2, feature.getOriginalHeight)
    meta.setValue(3, feature.getOriginalWidth)
    meta.setValue(4, 3) // channel
    val windows = feature[BoundingBox](ImageFeature.boundingBox)
    meta.setValue(5, windows.x1)
    meta.setValue(6, windows.y1)
    meta.setValue(7, windows.x2)
    meta.setValue(8, windows.y2)
    feature(ImageMeta.imageMeta) = meta
  }
}

object ImageMeta {
  val imageMeta = "ImageMeta"
  def apply(numClass: Int): ImageMeta = new ImageMeta(numClass)
}
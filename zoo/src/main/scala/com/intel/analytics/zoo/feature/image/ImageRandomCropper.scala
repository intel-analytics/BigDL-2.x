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

package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.dataset.image.{CropRandom, CropperMethod}
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.augmentation.RandomCropper

class ImageRandomCropper(cropWidth: Int, cropHeight: Int,
                         mirror: Boolean, cropperMethod: CropperMethod = CropRandom,
                         channels: Int = 3) extends ImageProcessing {

  private val internalRandomCropper = InternalRandomCropper(cropWidth, cropHeight,
    mirror, cropperMethod, channels)
  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    internalRandomCropper.apply(prev)
  }

  override def transformMat(feature: ImageFeature): Unit = {
    internalRandomCropper.transformMat(feature)
  }
}

object ImageRandomCropper {
  def apply(cropWidth: Int, cropHeight: Int,
            mirror: Boolean, cropperMethod: CropperMethod = CropRandom,
            channels: Int = 3): ImageRandomCropper =
    new ImageRandomCropper(cropWidth, cropHeight, mirror, cropperMethod, channels)
}

// transformMat in BigDL RandomCropper is protected and can't be directly accessed.
class InternalRandomCropper(cropWidth: Int, cropHeight: Int,
                            mirror: Boolean, cropperMethod: CropperMethod = CropRandom,
                            channels: Int = 3)
  extends RandomCropper(cropWidth, cropHeight, mirror, cropperMethod, channels) {

  override def transformMat(feature: ImageFeature): Unit = {
    super.transformMat(feature)
  }
}

object InternalRandomCropper {
  def apply(cropWidth: Int, cropHeight: Int,
            mirror: Boolean, cropperMethod: CropperMethod = CropRandom,
            channels: Int = 3): InternalRandomCropper =
    new InternalRandomCropper(cropHeight, cropWidth, mirror, cropperMethod, channels)
}

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

import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.transform.vision.image.MatToFloats
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat

import scala.reflect.ClassTag

class ImageMatToFloats(validHeight: Int, validWidth: Int, validChannels: Int,
                       outKey: String = ImageFeature.floats, shareBuffer: Boolean = true)
  extends ImageProcessing {
  @transient private var data: Array[Float] = _

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map(transform(_))
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    var input: OpenCVMat = null
    val (height, width, channel) = if (feature.isValid) {
      input = feature.opencvMat()
      (input.height(), input.width(), input.channels())
    } else {
      (validHeight, validWidth, validChannels)
    }
    if (!shareBuffer || null == data || data.length < height * width * channel) {
      data = new Array[Float](height * width * channel)
    }
    if (feature.isValid) {
      try {
        OpenCVMat.toFloatPixels(input, data)
      } finally {
        if (null != input) input.release()
        feature(ImageFeature.mat) = null
      }
    }
    feature(outKey) = data
    feature(ImageFeature.size) = (height, width, channel)
    feature
  }
}

object ImageMatToFloats {

  def apply[T: ClassTag](validHeight: Int = 300, validWidth: Int = 300, validChannels: Int = 3,
    outKey: String = ImageFeature.floats, shareBuffer: Boolean = true): ImageMatToFloats =
    new ImageMatToFloats(validHeight, validWidth, validChannels, outKey, shareBuffer)

}

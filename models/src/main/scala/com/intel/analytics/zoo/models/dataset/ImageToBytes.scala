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
package com.intel.analytics.zoo.models.dataset

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.utils.File
import com.intel.analytics.zoo.transform.vision.image.ImageFeature

class ImageToBytes extends Transformer[String, ImageFeature]{
  override def apply(prev: Iterator[String]): Iterator[ImageFeature] = {
    prev.map(path => {
      val rawImg = File.readBytes(path)
      val feature = ImageFeature()
      feature(ImageFeature.bytes) = rawImg
      feature("ImgPath") = path
      feature
    })
  }
}

object ImageToBytes {
  def apply(): ImageToBytes = new ImageToBytes()
}

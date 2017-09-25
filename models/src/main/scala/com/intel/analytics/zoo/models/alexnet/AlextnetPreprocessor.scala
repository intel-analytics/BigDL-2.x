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

package com.intel.analytics.zoo.models.alexnet

import com.intel.analytics.bigdl.example.loadmodel.AlexNetPreprocessor.imageSize
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.models.Preprocessor
import com.intel.analytics.zoo.models.dataset._
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, ImageFeature}
import com.intel.analytics.zoo.transform.vision.image.augmentation.{CenterCrop, PixelNormalizer, Resize}

@SerialVersionUID(-8096112278659979261L)
class AlexnetPreprocessor(mean : Tensor[Float]) extends Preprocessor {

  val transformer = ImageToMate() -> BytesToMat() -> Resize(256 , 256) ->
     PixelNormalizer(mean) -> CenterCrop(imageSize, imageSize) -> MateToSample(false)

  override def preprocess(path: Iterator[String]): Iterator[ImageSample] = {
    transformer(path)
  }
}

object AlexnetPreprocessor {
  def apply(mean: Tensor[Float]): AlexnetPreprocessor = new AlexnetPreprocessor(mean)
}
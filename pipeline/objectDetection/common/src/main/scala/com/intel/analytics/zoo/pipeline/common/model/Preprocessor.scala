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

package com.intel.analytics.zoo.pipeline.common.model

import com.intel.analytics.zoo.transform.vision.image.augmentation.{ChannelNormalize, Resize}
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, ImageFrame, MatToFloats}

object Preprocessor {
  def preprocessSsdVgg300(rdd: ImageFrame): ImageFrame = {
    preprocessSsd(rdd, 300, (123f, 117f, 104f), 1f)
  }

  def preprocessSsdVgg(rdd: ImageFrame, resolution: Int, nPartition: Int): ImageFrame = {
    preprocessSsd(rdd, resolution, (123f, 117f, 104f), 1f)
  }

  def preprocessSsd(imageFrame: ImageFrame, resolution: Int, meansRGB: (Float, Float, Float),
    scale: Float,
    batchPerPartition: Int = 1): ImageFrame = {
    imageFrame ->
      BytesToMat() ->
      Resize(resolution, resolution) ->
      ChannelNormalize(meansRGB._1, meansRGB._2, meansRGB._3, scale, scale, scale) ->
      MatToFloats(validHeight = resolution, validWidth = resolution)
  }
}

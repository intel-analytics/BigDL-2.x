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

package com.intel.analytics.zoo.feature.core.image

import java.util

import com.intel.analytics.zoo.feature.core.util.MatWrapper
import org.opencv.core.{Core, CvType, Mat, Scalar}

/**
 * image channel normalize
 * @param meanR
 * @param meanG
 * @param meanB
 */
class Normalize(meanR: Int, meanG: Int, meanB: Int)
  extends FeatureTransformer {
  override def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean = {
    Normalize.transform(input, output, meanR, meanG, meanB)
  }
}

object Normalize {
  def apply(mean: (Int, Int, Int)): Normalize = {
    new Normalize(mean._1, mean._2, mean._3)
  }

  def transform(input: MatWrapper, output: MatWrapper,
    meanR: Int, meanG: Int, meanB: Int): Boolean = {
    if (input.`type`() != CvType.CV_32FC3) {
      input.convertTo(input, CvType.CV_32FC3)
    }
    val inputChannels = new util.ArrayList[Mat]()
    Core.split(input, inputChannels)
    require(inputChannels.size() == 3)
    val outputChannels = if (output != input) {
      output.create(input.rows(), input.cols(), input.`type`())
      val channels = new util.ArrayList[Mat]()
      Core.split(output, channels)
      channels
    } else inputChannels

    Core.subtract(inputChannels.get(0), new Scalar(meanB), outputChannels.get(0))
    Core.subtract(inputChannels.get(1), new Scalar(meanG), outputChannels.get(1))
    Core.subtract(inputChannels.get(2), new Scalar(meanR), outputChannels.get(2))
    Core.merge(outputChannels, output)

    (0 until inputChannels.size()).foreach(inputChannels.get(_).release())
    if (input != output) {
      (0 until outputChannels.size()).foreach(outputChannels.get(_).release())
    }
    true
  }
}

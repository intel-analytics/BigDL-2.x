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

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.zoo.feature.core.util.MatWrapper
import org.opencv.core.{Core, Mat}
import org.opencv.imgproc.Imgproc

/**
 * Adjust image saturation
 * @param delta
 */
class Saturation(delta: Float)
  extends FeatureTransformer {
  override def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean = {
    Saturation.transform(input, output, delta)
    true
  }
}

object Saturation {
  def apply(delta: Float): Saturation = new Saturation(delta)

  def transform(input: MatWrapper, output: MatWrapper, delta: Float): MatWrapper = {
    if (Math.abs(delta - 1) != 1e-3) {
      // Convert to HSV colorspace
      Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2HSV)

      // Split the image to 3 channels.
      val channels = new util.ArrayList[Mat]()
      Core.split(output, channels)

      // Adjust the saturation.
      channels.get(1).convertTo(channels.get(1), -1, delta, 0)
      Core.merge(channels, output)
      (0 until channels.size()).foreach(channels.get(_).release())
      // Back to BGR colorspace.
      Imgproc.cvtColor(output, output, Imgproc.COLOR_HSV2BGR)
    } else {
      if (input != output) input.copyTo(output)
    }
    output
  }
}

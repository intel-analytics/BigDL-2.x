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

package com.intel.analytics.dlfeature.core.image

import java.util

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.dlfeature.core.util.MatWrapper
import org.opencv.core.{Core, Mat}
import org.opencv.imgproc.Imgproc

class Hue(delta: Float)
  extends ImageTransformer {
  override def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean = {
    Hue.transform(input, output, delta)
    true
  }
}

object Hue {
  def apply(delta: Float): Hue = new Hue(delta)

  def transform(input: MatWrapper, output: MatWrapper, delta: Float): MatWrapper = {
    if (delta != 0) {
      // Convert to HSV colorspae
      Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2HSV)

      // Split the image to 3 channels.
      val channels = new util.ArrayList[Mat]()
      Core.split(output, channels)

      // Adjust the hue.
      channels.get(0).convertTo(channels.get(0), -1, 1, delta)
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

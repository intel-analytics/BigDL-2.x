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

package com.intel.analytics.zoo.transform.vision.image.augmentation

import java.util

import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.zoo.transform.vision.util.NormalizedBox
import org.opencv.core.{Core, Mat, Rect, Scalar}


class Expand(meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
                  maxExpandRatio: Double = 4.0)
  extends FeatureTransformer {

  var expandMat: OpenCVMat = _

  def transform(input: OpenCVMat,
                output: OpenCVMat): NormalizedBox = {
    val imgHeight = input.rows()
    val imgWidth = input.cols()
    val expandRatio = RNG.uniform(1, maxExpandRatio)
    val height = (imgHeight * expandRatio).toInt
    val width = (imgWidth * expandRatio).toInt
    val hOff = RNG.uniform(0, height - imgHeight).floor.toFloat
    val wOff = RNG.uniform(0, width - imgWidth).floor.toFloat
    val expandBbox = new NormalizedBox()
    expandBbox.x1 = -wOff / imgWidth
    expandBbox.y1 = -hOff / imgHeight
    expandBbox.x2 = (width - wOff) / imgWidth
    expandBbox.y2 = (height - hOff) / imgHeight
    val bboxRoi = new Rect(wOff.toInt, hOff.toInt, imgWidth.toInt, imgHeight.toInt)

    output.create(height, width, input.`type`())

    // Split the image to 3 channels.
    val channels = new util.ArrayList[Mat]()
    Core.split(output, channels)
    require(channels.size() == 3)
    channels.get(0).setTo(new Scalar(meansB))
    channels.get(1).setTo(new Scalar(meansG))
    channels.get(2).setTo(new Scalar(meansR))
    Core.merge(channels, output)
    input.copyTo(output.submat(bboxRoi))
    // release memory
    (0 to 2).foreach(channels.get(_).release())
    expandBbox
  }

  override def transformMat(prev: ImageFeature): Unit = {
    val mat = prev.opencvMat()
    if (Math.abs(maxExpandRatio - 1) >= 1e-2) {
      if (null == expandMat) expandMat = new OpenCVMat()
      val expandBbox = transform(mat, expandMat)
      expandMat.copyTo(mat)
      if (prev.hasLabel()) {
        prev(ImageFeature.expandBbox) = expandBbox
      }
      if (null != expandMat) expandMat.release()
    }
  }
}

object Expand {
  def apply(meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
            maxExpandRatio: Double = 4.0): Expand =
    new Expand(meansR, meansG, meansB, maxExpandRatio)
}



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

package com.intel.analytics.zoo.dlfeature.core.image

import java.util

import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.zoo.dlfeature.core.label.LabelTransformMethod
import com.intel.analytics.zoo.dlfeature.core.util.{MatWrapper, NormalizedBox}
import org.opencv.core.{Core, Mat, Rect, Scalar}

import scala.util.Random

class Expand(expandProb: Double = 0.5, maxExpandRatio: Double = 4.0,
  meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
  labelTransformMethod: Option[LabelTransformMethod] = None)
  extends ImageTransformer {

  if (labelTransformMethod.isDefined) setLabelTransfomer(labelTransformMethod.get)
  var expandMat: MatWrapper = _

  override def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean = {
    if (Math.abs(maxExpandRatio - 1) < 1e-2 || Random.nextFloat() > expandProb) {
      if (input != output) input.copyTo(output)
      false
    } else {
      if (null == expandMat) expandMat = new MatWrapper()
      val expandRatio = RNG.uniform(1, maxExpandRatio)
      val expandBbox = Expand.transform(input, expandRatio, expandMat, meansR, meansG, meansB)
      expandMat.copyTo(output)
      if (feature.hasLabel()) {
        feature("expandBbox") = expandBbox
      }
      feature(Feature.height) = output.height()
      feature(Feature.width) = output.width()
      if (null != expandMat) expandMat.release()
      true
    }
  }
}

object Expand {
  def apply(expandProb: Double = 0.5, maxExpandRatio: Double = 4.0,
    meansR: Int = 123, meansG: Int = 117, meansB: Int = 104,
    labelTransformMethod: Option[LabelTransformMethod] = None): Expand =
    new Expand(expandProb, maxExpandRatio, meansR, meansG, meansB, labelTransformMethod)

  def transform(input: MatWrapper, expandRatio: Double,
    output: MatWrapper,
    meansR: Int = 123, meansG: Int = 117, meansB: Int = 104): NormalizedBox = {
    val imgHeight = input.rows()
    val imgWidth = input.cols()
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
}



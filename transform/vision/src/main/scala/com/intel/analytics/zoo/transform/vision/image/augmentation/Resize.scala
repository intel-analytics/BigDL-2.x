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

import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.transform.vision.image.{FeatureTransformer, ImageFeature}
import org.apache.log4j.Logger
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

import scala.util.Random

/**
 *
 * @param resizeH
 * @param resizeW
 * @param resizeMode if resizeMode = -1, random select a mode from
 * (Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
 *                   Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)
 */
class Resize(resizeH: Int, resizeW: Int,
  resizeMode: Int = Imgproc.INTER_LINEAR)
  extends FeatureTransformer {

  private val interpMethods = Array(Imgproc.INTER_LINEAR, Imgproc.INTER_CUBIC, Imgproc.INTER_AREA,
    Imgproc.INTER_NEAREST, Imgproc.INTER_LANCZOS4)

  override def transformMat(feature: ImageFeature): Unit = {
    val interpMethod = if (resizeMode == -1) {
      interpMethods(new Random().nextInt(interpMethods.length))
    } else {
      resizeMode
    }
    Resize.transform(feature.opencvMat(), feature.opencvMat(), resizeW, resizeH, interpMethod)
  }
}

object Resize {
  val logger = Logger.getLogger(getClass)

  def apply(resizeH: Int, resizeW: Int,
    resizeMode: Int = Imgproc.INTER_LINEAR): Resize =
    new Resize(resizeH, resizeW, resizeMode)

  def transform(input: OpenCVMat, output: OpenCVMat, resizeW: Int, resizeH: Int,
                mode: Int = Imgproc.INTER_LINEAR)
  : OpenCVMat = {
    Imgproc.resize(input, output, new Size(resizeW, resizeH), 0, 0, mode)
    output
  }
}

/**
 * resize the image by randomly choosing a scale
 * @param scales array of scale options that for random choice
 * @param scaleMultipleOf Resize test images so that its width and height are multiples of
 * @param maxSize Max pixel size of the longest side of a scaled input image
 */
case class RandomResize(scales: Array[Int], scaleMultipleOf: Int = 1,
  maxSize: Float = 1000f) extends FeatureTransformer {

  override def transformMat(feature: ImageFeature): Unit = {
    val scaleTo = scales(Random.nextInt(scales.length))
    val (height, width) = getWidthHeightAfterRatioScale(feature.opencvMat(), scaleTo)
    Resize.transform(feature.opencvMat(), feature.opencvMat(), width, height)
  }

  /**
   * get the width and height of scaled image
   * @param img original image
   */
  def getWidthHeightAfterRatioScale(img: OpenCVMat, scaleTo: Float): (Int, Int) = {
    val imSizeMin = Math.min(img.width(), img.height())
    val imSizeMax = Math.max(img.width(), img.height())
    var imScale = scaleTo.toFloat / imSizeMin.toFloat
    // Prevent the biggest axis from being more than MAX_SIZE
    if (Math.round(imScale * imSizeMax) > maxSize) {
      imScale = maxSize / imSizeMax.toFloat
    }

    val imScaleH = (Math.floor(img.height() * imScale / scaleMultipleOf) *
      scaleMultipleOf / img.height()).toFloat
    val imScaleW = (Math.floor(img.width() * imScale / scaleMultipleOf) *
      scaleMultipleOf / img.width()).toFloat
    val width = imScaleW * img.width()
    val height = imScaleH * img.height()
    (height.toInt, width.toInt)
  }
}


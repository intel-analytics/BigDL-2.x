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

package com.intel.analytics.zoo.transform.vision.image

import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD


class BytesToMat()
  extends FeatureTransformer {
  override def transform(feature: ImageFeature): ImageFeature = {
    if (!feature.isValid) return feature
    val bytes = feature(ImageFeature.bytes).asInstanceOf[Array[Byte]]
    var mat: OpenCVMat = null
    try {
      mat = OpenCVMat.toMat(bytes)
      feature(ImageFeature.mat) = mat
      feature(ImageFeature.originalW) = mat.width()
      feature(ImageFeature.originalH) = mat.height()
    } catch {
      case e: Exception =>
        e.printStackTrace()
        feature(ImageFeature.originalW) = -1
        feature(ImageFeature.originalH) = -1
        feature.isValid = false
    }
    feature
  }
}

object BytesToMat {
  def apply(): BytesToMat = new BytesToMat()
}


class MatToFloats(validHeight: Int, validWidth: Int,
  meanRGB: Option[(Int, Int, Int)] = None, outKey: String = ImageFeature.floats)
  extends FeatureTransformer {
  @transient private var data: Array[Float] = _
  @transient private var floatMat: OpenCVMat = _

  private def normalize(img: Array[Float], meanR: Int, meanG: Int, meanB: Int): Array[Float] = {
    val content = img
    require(content.length % 3 == 0)
    var i = 0
    while (i < content.length) {
      content(i + 2) = content(i + 2) - meanR
      content(i + 1) = content(i + 1) - meanG
      content(i + 0) = content(i + 0) - meanB
      i += 3
    }
    img
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    var input: OpenCVMat = null
    val (height, width) = if (feature.isValid) {
      input = feature.opencvMat()
      (input.height(), input.width())
    } else {
      (validHeight, validWidth)
    }
    if (null == data || data.length < height * width * 3) {
      data = new Array[Float](height * width * 3)
    }
    if (feature.isValid) {
      try {
        if (floatMat == null) {
          floatMat = new OpenCVMat()
        }
        OpenCVMat.toFloatBuf(input, data, floatMat)
        if (meanRGB.isDefined) {
          normalize(data, meanRGB.get._1, meanRGB.get._2, meanRGB.get._3)
        }
      } finally {
        if (null != input) input.release()
      }
    }
    feature(outKey) = data
    feature(ImageFeature.width) = width
    feature(ImageFeature.height) = height
    feature
  }
}

object MatToFloats {
  val logger = Logger.getLogger(getClass)

  def apply(validHeight: Int = 300, validWidth: Int = 300,
    meanRGB: Option[(Int, Int, Int)] = None, outKey: String = ImageFeature.floats): MatToFloats =
    new MatToFloats(validHeight, validWidth, meanRGB, outKey)
}

case class ImageFeatureRdd(rdd: RDD[ImageFeature]) {

}

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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.transform.vision.image.ImageFeature
import org.opencv.core.CvType


class MateToTensor(toRGB: Boolean = true) extends Transformer[ImageFeature, Tensor[Float]] {

  override def apply(prev: Iterator[ImageFeature]): Iterator[Tensor[Float]] = {
    prev.map(feature => {
      val openCVMat = feature.opencvMat()
      if (openCVMat.`type`() != CvType.CV_32FC3) {
        openCVMat.convertTo(openCVMat, CvType.CV_32FC3)
      }
      val floatContent = new Array[Float](openCVMat.height() * openCVMat.width() * 3)
      val height = openCVMat.height()
      val width = openCVMat.width()
      openCVMat.get(0, 0, floatContent)

      val tensorArray = new Array[Float](floatContent.length)

      copyTo(floatContent, tensorArray, toRGB)

      Tensor[Float](tensorArray, Array(1, 3, height, width))
    })
  }

  private def copyTo(from: Array[Float],to: Array[Float], toRGB: Boolean = true): Unit = {
    val frameLength = from.length / 3
    var j = 0
    if(toRGB) {
      while (j < frameLength) {
        to(j) = from(j * 3 + 2)
        to(j + frameLength) = from(j * 3 + 1)
        to(j + frameLength * 2) = from(j * 3)
        j += 1
      }
    } else {
      while (j < frameLength) {
        to(j) = from(j * 3)
        to(j + frameLength) = from(j * 3 + 1)
        to(j + frameLength * 2) = from(j * 3 + 2)
        j += 1
      }
    }
  }
}

object MateToTensor {
  def apply(toRGB: Boolean = true): MateToTensor = new MateToTensor(toRGB)
  /*
  //From NHWC to NCHW
  def bGRToFloatTensor(feature: ImageFeature, toRGB: Boolean = true) : Tensor[Float] = {

    val openCVMat = feature.opencvMat()
    if (openCVMat.`type`() != CvType.CV_32FC3) {
      openCVMat.convertTo(openCVMat, CvType.CV_32FC3)
    }
    val floatContent = new Array[Float](openCVMat.height() * openCVMat.width() * 3)
    val height = openCVMat.height()
    val width = openCVMat.width()
    openCVMat.get(0, 0, floatContent)

    val tensorArray = new Array[Float](floatContent.length)

    copyTo(floatContent, tensorArray, toRGB)

    Tensor[Float](tensorArray, Array(1, 3, height, width))
  }

  private def copyTo(from: Array[Float],to: Array[Float], toRGB: Boolean = true): Unit = {
    val frameLength = from.length / 3
    var j = 0
    if(toRGB) {
      while (j < frameLength) {
        to(j) = from(j * 3 + 2)
        to(j + frameLength) = from(j * 3 + 1)
        to(j + frameLength * 2) = from(j * 3)
        j += 1
      }
    } else {
      while (j < frameLength) {
        to(j) = from(j * 3)
        to(j + frameLength) = from(j * 3 + 1)
        to(j + frameLength * 2) = from(j * 3 + 2)
        j += 1
      }
    }
  }
  */
}

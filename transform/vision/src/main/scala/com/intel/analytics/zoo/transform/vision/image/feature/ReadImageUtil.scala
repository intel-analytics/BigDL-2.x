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

package com.intel.analytics.zoo.transform.vision.image.feature

import com.intel.analytics.zoo.transform.vision.image.opencv.{OpenCV, OpenCVMat}
import org.opencv.core.{MatOfByte, Size}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

object ReadImageUtil extends Serializable {

  OpenCV.loadIfNecessary()

  /**
   * @param fileContent bytes from an image file.
   * @return OpenCV Mat
   */
  private[vision] def readImageAsMat(fileContent: Array[Byte]): OpenCVMat = {
    val mat = Imgcodecs.imdecode(new MatOfByte(fileContent: _*), Imgcodecs.CV_LOAD_IMAGE_COLOR)
    new OpenCVMat(mat)
  }

  /**
   * @param fileContent bytes representation for an image file.
   * @param smallSideSize the size of the smallest side after resize
   * @return OpenCV Mat
   */
  private[vision] def readImageAsMat(fileContent: Array[Byte], smallSideSize: Int): OpenCVMat = {
    aspectPreseveRescale(readImageAsMat(fileContent), smallSideSize)
  }

  /**
   * @param fileContent bytes representation for an image file.
   * @return (data, height, width, numChannel)
   */
  def readImageAsBytes(fileContent: Array[Byte]): (Array[Byte], Int, Int, Int) = {
    try {
      val mat = readImageAsMat(fileContent)
      mat2Bytes(mat)
    } catch {
      case e: Exception =>
        println(e)
        null
    }
  }

  /**
   * @param fileContent bytes representation for an image file.
   * @param smallSideSize the size of the smallest side after resize
   * @return (data, height, width, numChannel)
   */
  def readImageAsBytes(
      fileContent: Array[Byte],
      smallSideSize: Int
      ): (Array[Byte], Int, Int, Int) = {
    try {
      val mat = aspectPreseveRescale(readImageAsMat(fileContent), smallSideSize)
      mat2Bytes(mat)
    } catch {
      case e: Exception =>
        println(e)
        null
    }
  }

  /**
   * @param fileContent bytes representation for an image file.
   * @param smallSideSize the size of the smallest side after resize
   * @param divisor divide each pixel by divisor. E.g. if divisor = 255, each pixel is in [0, 1]
   * @return (data, height, width, numChannel)
   */
  def readImageAsFloats(
      fileContent: Array[Byte],
      smallSideSize: Int,
      divisor: Float = 1.0f): (Array[Float], Int, Int, Int) = {
    try {
      val (bytes, h, w, c) = readImageAsBytes(fileContent, smallSideSize)
      val floats = bytes.map(b => (b & 0xff) / divisor)
      (floats, h, w, c)
    } catch {
      case e: Exception =>
        println(e)
        null
    }
  }

  private[vision] def mat2Bytes(mat: OpenCVMat): (Array[Byte], Int, Int, Int) = {
    val w = mat.width()
    val h = mat.height()
    val c = mat.channels()
    val bytes = Array.ofDim[Byte](c * w * h)
    mat.get(0, 0, bytes)
    (bytes, h, w, c)
  }

  private[vision] def aspectPreseveRescale(srcMat: OpenCVMat, smallSideSize: Int): OpenCVMat = {
    val origW = srcMat.width()
    val origH = srcMat.height()
    val (resizeW, resizeH) = if (origW < origH) {
      (smallSideSize, origH * smallSideSize  / origW)
    } else {
      (origW * smallSideSize / origH, smallSideSize)
    }
    val dst = srcMat
    Imgproc.resize(srcMat, dst, new Size(resizeW, resizeH), 0, 0, Imgproc.INTER_LINEAR)
    dst
  }
}




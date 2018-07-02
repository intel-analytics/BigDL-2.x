/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.feature.image

import com.intel.analytics.bigdl.opencv.OpenCV
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat

import org.opencv.core.{Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs

private[zoo] object OpenCVMethod {
  OpenCV.isOpenCVLoaded

  /**
   * convert image file in bytes to opencv mat with BGR
   *
   * @param fileContent bytes from an image file
   * @param flags specifying the color type of a loaded image, same as in OpenCV.imread.
   *              By default is Imgcodecs.CV_LOAD_IMAGE_COLOR and returns a 3-channel color image
   * @return opencv mat
   */
  def fromImageBytes(fileContent: Array[Byte],
                     flags: Int = Imgcodecs.CV_LOAD_IMAGE_COLOR): OpenCVMat = {
    var mat: Mat = null
    var matOfByte: MatOfByte = null
    var result: OpenCVMat = null
    try {
      matOfByte = new MatOfByte(fileContent: _*)
      mat = Imgcodecs.imdecode(matOfByte, flags)
      result = new OpenCVMat(mat)
    } catch {
      case e: Exception =>
        if (null != result) result.release()
        throw e
    } finally {
      if (null != mat) mat.release()
      if (null != matOfByte) matOfByte.release()
    }
    result
  }
}

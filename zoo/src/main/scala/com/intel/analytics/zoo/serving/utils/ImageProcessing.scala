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

package com.intel.analytics.zoo.utils

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.feature.image.OpenCVMethod
import org.opencv.imgcodecs.Imgcodecs

object ImageProcessing {

  def bytesToBGRTensor(bytes: Array[Byte], chwFlag: Boolean = true): Tensor[Float] = {
    val mat = OpenCVMethod.fromImageBytes(bytes, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())

    val data = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, data)

    val imageTensor = Tensor[Float](data, Array(height, width, channel))

    if (chwFlag) {
      imageTensor.transpose(1, 3).transpose(2, 3).contiguous()
    } else {
      imageTensor
    }

  }
}

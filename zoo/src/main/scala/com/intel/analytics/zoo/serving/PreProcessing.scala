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

package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.feature.image.OpenCVMethod
import com.intel.analytics.zoo.serving.DataType.DataType
import org.apache.spark.bigdl.api.python.BigDLSerDe
import org.opencv.imgcodecs.Imgcodecs

class PreProcessing(s: String) {
  def bytesToTensor(dataType: String, chwFlag: Boolean = true): Tensor[Float] = {
    val b = java.util.Base64.getDecoder.decode(s)

    val result = dataType match {
      case "image" =>
        val mat = OpenCVMethod.fromImageBytes(b, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
        val (height, width, channel) = (mat.height(), mat.width(), mat.channels())

        val data = new Array[Float](height * width * channel)
        OpenCVMat.toFloatPixels(mat, data)

        val imageTensor = Tensor[Float](data, Array(height, width, channel))

        if (chwFlag) {
        imageTensor.transpose(1, 3).transpose(2, 3).contiguous()
        } else {
        imageTensor
        }
      case "tensor" =>
        val tensorLoad = BigDLSerDe.loads(b)
//        if (tensorLoad.isInstanceOf[JTensor]) {
        val t = PythonBigDL.ofFloat().toTensor(tensorLoad.asInstanceOf[JTensor])
        t
//        }
    }

    result
  }
}
object PreProcessing {
  def apply(s: String, dataType: String = "image", chwFlag: Boolean = true,
            args: Array[Int] = Array()): Tensor[Float] = {
    val cls = new PreProcessing(s)
    val t = cls.bytesToTensor(dataType, chwFlag)
    for (op <- args) {
      // new processing features add to here
    }
    t
  }
}

object DataType extends Enumeration {
  type DataType = Value

  val IMAGE = Value("Image")
  val TENSOR = Value("Tensor")
}
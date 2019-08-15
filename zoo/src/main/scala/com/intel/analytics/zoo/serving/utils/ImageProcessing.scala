package com.intel.analytics.zoo.utils

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.feature.image.OpenCVMethod
import org.opencv.imgcodecs.Imgcodecs

object ImageProcessing {

  def bytesToBGRTensor(bytes: Array[Byte]): Tensor[Float] = {
    val mat = OpenCVMethod.fromImageBytes(bytes, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)

    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())
    val data = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, data)
    val imageTensor: Tensor[Float] = Tensor[Float]()
    imageTensor.resize(channel, height, width)
    val storage = imageTensor.storage().array()
    imageTensor.transpose(1, 2).transpose(2, 3)
    val offset = 0
    val frameLength = width * height
    var j = 0
    while (j < frameLength) {
      storage(offset + j) = data(j * 3)
      storage(offset + j + frameLength) = data(j * 3 + 1)
      storage(offset + j + frameLength * 2) = data(j * 3 + 2)
      j += 1
    }
    imageTensor
  }
}

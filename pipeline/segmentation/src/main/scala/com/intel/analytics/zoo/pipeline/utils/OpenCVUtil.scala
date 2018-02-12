package com.intel.analytics.zoo.pipeline.utils

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import org.opencv.core.Mat

object OpenCVUtil {

  def toTensor(input: Mat): Tensor[Float] = {
    val floats = OpenCVMat.toFloatPixels(input)
    Tensor(Storage(floats._1)).resize(input.height(), input.width(), input.channels())
  }
}

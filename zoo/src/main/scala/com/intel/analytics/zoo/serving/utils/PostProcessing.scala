package com.intel.analytics.zoo.serving.utils

import com.intel.analytics.bigdl.tensor.Tensor

object PostProcessing {
  def getInfofromTensor(outputSize: Int, tensor: Tensor[Float]): String = {
    val output = TensorUtils.getTopN(outputSize, tensor)
    var value: String = "{"
    (0 until outputSize - 1).foreach( j => {
      val tmpValue = "\"" + output(j)._1 + "\":\"" +
        output(j)._2.toString + "\","
      value += tmpValue
    })
    value += "\"" + output(outputSize - 1)._1 + "\":\"" +
      output(outputSize - 1)._2.toString
    value += "\"}"
    value
  }
}

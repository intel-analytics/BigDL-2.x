package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

class ImageProcesser(bytes: Array[Byte], cropWidth: Int, cropHeight: Int, meanR: Int, meanG: Int, meanB: Int, scale: Double) extends ImageProcessing {
  def preProcess(bytes: Array[Byte], cropWidth: Int, cropHeight: Int, meanR: Int, meanG: Int, meanB: Int, scale: Double) = {
    val imageMatt = byteArrayToMat(bytes)
    val imageCenti = centerCrop(imageMatt, cropWidth, cropHeight)
    val imageTensors = matToNCHWAndRGBTensor(imageCenti)
    val norImaTensors = channelScaledNormalize(imageTensors, meanR, meanG, meanB, scale)
 //   imageTensors
    norImaTensors
  }
}

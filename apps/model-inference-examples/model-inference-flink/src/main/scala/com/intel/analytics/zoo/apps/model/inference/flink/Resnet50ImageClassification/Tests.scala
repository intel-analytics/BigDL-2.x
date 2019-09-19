package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

import java.io.{File, FileInputStream}
import java.util

import com.intel.analytics.zoo.feature.common.ArrayToTensor
import com.intel.analytics.zoo.pipeline.inference.{InferenceSupportive, JTensor}
import org.apache.commons.io.FileUtils

import scala.collection.JavaConverters._
import scala.io.Source

object Tests extends App with InferenceSupportive {

  ////////////loading and processing images
  val imageFolder = new File("/home/joy/Data/pic")
  val fileList = imageFolder.listFiles.toList
  fileList.foreach(println)

  val inputs =  fileList.map(file => {
    val imageBytes = FileUtils.readFileToByteArray(file)
    val imageProcess = new ImageProcessor
    val res = imageProcess.preProcess(imageBytes, 224, 224, 123, 116, 103, 1.0)
    val input = new JTensor(res, Array(1, 224, 224, 3))
    List(util.Arrays.asList(input)).asJava
  })
  inputs.foreach(println)

  ///////////model_params
  var modelType = "resnet_v1_50"
  var checkpointPath: String = "/home/joy/models/resnet_v1_50.ckpt"
  var ifReverseInputChannels = true
  var inputShape = Array(1, 224, 224, 3)
  var meanValues = Array(123.68f, 116.78f, 103.94f)
  var scale = 1.0f
  val classLoader = this.getClass.getClassLoader
  val fileSize = new File(checkpointPath).length()
  val inputStream = new FileInputStream(checkpointPath)
  val modelBytes = new Array[Byte](fileSize.toInt)
  inputStream.read(modelBytes)
  val labels = Source.fromFile("/home/joy/analytics-zoo/zoo/src/main/resources/imagenet_classname.txt").getLines.toList

  val resnet50InferenceModel = new Resnet50InferenceModel(1, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)

  inputs.map(input => {
    val result = resnet50InferenceModel.doPredict(input)
    val data = result.get(0).get(0).getData
    val max: Float = data.max
    val index = data.indexOf(max)
    val label = labels(index)
    println(index+label)
  })
}




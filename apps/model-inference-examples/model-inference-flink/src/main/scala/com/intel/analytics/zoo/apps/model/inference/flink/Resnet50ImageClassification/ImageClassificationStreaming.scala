package com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification

import java.io.{File, FileInputStream}
import java.util.{Arrays, List => JList}

import com.intel.analytics.zoo.pipeline.inference.JTensor
import org.apache.commons.io.FileUtils
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.datastream.DataStreamUtils
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import scala.io.Source

object ImageClassificationStreaming {

  def main(args: Array[String]): Unit = {
    var modelType = "resnet_v1_50"
    var checkpointPath: String = "/home/joy/models/resnet_v1_50.ckpt"
    var ifReverseInputChannels = true
    var inputShape = Array(1, 224, 224, 3)
    var meanValues = Array(123.68f, 116.78f, 103.94f)
    var scale = 1.0f
//    val lines=Source.fromFile("/home/joy/Data/classes.txt" ).getLines.toList
//    println(lines)

    try {
      val params = ParameterTool.fromArgs(args)
      modelType = params.get("modelType")
      checkpointPath = params.get("checkpointPath")
      inputShape = if (params.has("inputShape")) {
        val inputShapeStr = params.get("inputShape")
        inputShapeStr.split(",").map(_.toInt).toArray
      } else Array(1, 224, 224, 3)
      ifReverseInputChannels = if (params.has("ifReverseInputChannels")) params.getBoolean("ifReverseInputChannels") else true
      meanValues = if (params.has("meanValues")) {
        val meanValuesStr = params.get("meanValues")
        meanValuesStr.split(",").map(_.toFloat).toArray
      } else Array(123.68f, 116.78f, 103.94f)
      scale = if (params.has("scale")) params.getFloat("scale") else 1.0f
    } catch {
      case e: Exception => {
        System.err.println("Please run 'ImageClassificationStreaming --modelType <modelType> --checkpointPath <checkpointPath> " +
          "--inputShape <inputShapes> --ifReverseInputChannels <ifReverseInputChannels> --meanValues <meanValues> --scale <scale>" +
          "--parallelism <parallelism>'.")
        return
      }
    }

    println("start ImageClassificationStreaming job...")
    println("params resolved", modelType, checkpointPath, inputShape.mkString(","), ifReverseInputChannels, meanValues.mkString(","), scale)

    val classLoader = this.getClass.getClassLoader

    val fileSize = new File(checkpointPath).length()
    val inputStream = new FileInputStream(checkpointPath)
    val modelBytes = new Array[Byte](fileSize.toInt)
    inputStream.read(modelBytes)

    val imageFolder = new File("/home/joy/Data/image2")
    val fileList = imageFolder.listFiles.toList

    println("fileList", fileList)

    var inputs = new ListBuffer[Array[Float]]()
    for (file <- fileList) {
      val imageBytes = FileUtils.readFileToByteArray(file)
      val imageProcess = new ImageProcesser(imageBytes, 224, 224, 123, 116, 103, 1.0)
      val res = imageProcess.preProcess(imageBytes, 224, 224, 123, 116, 103, 1.0)
      println("imageTensor",res)
      val inputImage = new Array[Float](res.nElement())
      println(inputImage)
      inputs += inputImage
    }

    val inputsList = inputs.toList
    println("inputsList", inputsList)

    val env: StreamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment
    println(env.getConfig)

    val dataStream: DataStream[Array[Float]] = env.fromCollection(inputsList)
    val tensorStream: DataStream[JList[JList[JTensor]]] = dataStream.map(value => {
      val input = new JTensor(value, Array(1, 224, 224, 3))
      println(input)
      val data = Arrays.asList(input)
      List(data).asJava
    })

    val resultStream = tensorStream.map(new ModelPredictionMapFunction(modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale))

    env.execute("ImageClassificationStreaming")

    val results = DataStreamUtils.collect(resultStream.javaStream).asScala

    println(" Printing result ...")
    results.foreach(println)
  }

}

class ModelPredictionMapFunction(modelType: String, modelBytes: Array[Byte], inputShape: Array[Int], ifReverseInputChannels: Boolean, meanValues: Array[Float], scale: Float) extends RichMapFunction[JList[JList[JTensor]], String] {
  var resnet50InferenceModel: Resnet50InferenceModel = _

  override def open(parameters: Configuration): Unit = {
    resnet50InferenceModel = new Resnet50InferenceModel(1, modelType, modelBytes, inputShape, ifReverseInputChannels, meanValues, scale)
  }

  override def close(): Unit = {
    resnet50InferenceModel.doRelease()
  }

  override def map(in: JList[JList[JTensor]]): (String) = {
    val lines=Source.fromFile("/home/joy/analytics-zoo/zoo/src/main/resources/imagenet_classname.txt" ).getLines.toList
    val outputData = resnet50InferenceModel.doPredict(in).get(0).get(0).getData
    val max: Float = outputData.max
    val index = outputData.indexOf(max)
    val label = lines(index+1)
    (label)
  }
}

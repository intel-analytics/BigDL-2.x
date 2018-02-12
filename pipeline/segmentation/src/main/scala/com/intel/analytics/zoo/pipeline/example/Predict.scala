package com.intel.analytics.zoo.pipeline.example

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.PaddingParam
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.utils.{Engine, File}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{AspectScale, ChannelNormalize, FixExpand}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.pipeline.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser

import scala.io.Source

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.pipeline.ssd").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class DemoParam(imageFolder: String = "",
    output: String = "",
    model: String = "",
    classname: String = "",
    quantize: Boolean = false)

  val parser = new OptionParser[DemoParam]("BigDL Object Segmentation Demo") {
    head("BigDL Object Segmentation Demo")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]('o', "output")
      .text("where you put the output image data")
      .action((x, c) => c.copy(output = x))
      .required()
    opt[String]("model")
      .text("BigDL model")
      .action((x, c) => c.copy(model = x))
    opt[String]("classname")
      .text("file store class name")
      .action((x, c) => c.copy(classname = x))
      .required()
    opt[Boolean]('q', "quantize")
      .text("whether to quantize")
      .action((x, c) => c.copy(quantize = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, DemoParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("BigDL Object Segmentation Demo")
      val sc = new SparkContext(conf)
      Engine.init
      var model = Module.loadModule[Float](params.model).evaluate()
      model = if (params.quantize) model.quantize() else model
      logger.info("load model done")
      val labelMap = LabelReader.readCocoLabelMap()

      val images = ImageFrame.read(params.imageFolder, sc) ->
        AspectScale(800, 1, 1024, useScaleFactor = false, minScale = Some(1)) ->
        FixExpand(1024, 1024) ->
        ChannelNormalize(103.9f, 116.8f, 123.7f) ->
        MatToTensor() -> ImageMeta(81) ->
        ImageFrameToSample(Array(ImageFeature.imageTensor, ImageMeta.imageMeta))
      val detectOut = UnmodeDetection()
      val visualizer = Visualizer(labelMap, 0.75f, encoding = "jpg")

      val output = model.predictImage(images,
        shareBuffer = false,
        batchPerPartition = 1).transform(detectOut)

      val visualized = visualizer(output).toDistributed()
      val result = visualized.rdd.map(imageFeature =>
        (imageFeature.uri(), imageFeature[Array[Byte]](Visualizer.visualized))).collect()

      result.foreach(x => {
        File.saveBytes(x._2, getOutPath(params.output, x._1, "jpg"), true)
      })
      logger.info(s"labeled images are saved to ${params.output}")
      sc.stop()
    }
  }

  def getOutPath(outPath: String, uri: String, encoding: String): String = {
    Paths.get(outPath,
      s"detection_${ uri.substring(uri.lastIndexOf("/") + 1,
        uri.lastIndexOf(".")) }.${encoding}").toString
  }
}


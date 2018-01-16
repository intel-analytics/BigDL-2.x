/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.pipeline.ssd.example

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.zoo.pipeline.common.IOUtils
import com.intel.analytics.zoo.pipeline.common.caffe.SSDCaffeLoader
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.RecordToFeature
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.zoo.models.Predictor
import com.intel.analytics.zoo.models.objectdetection.utils.{ObjectDetectionConfig, ScaleDetection}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.models.Configure
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.io.Source

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.pipeline.ssd").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PascolVocDemoParam(imageFolder: String = "",
    folderType: String = "local",
    model: Option[String] = None,
    caffeDefPath: Option[String] = None,
    caffeModelPath: Option[String] = None,
    batchPerPartition: Int = 2,
    classname: String = "",
    resolution: Int = 300,
    nPartition: Int = 1,
    quantize: Boolean = false)

  val parser = new OptionParser[PascolVocDemoParam]("BigDL SSD Demo") {
    head("BigDL SSD Demo")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]("folderType")
      .text("local image folder or hdfs sequence folder")
      .action((x, c) => c.copy(folderType = x))
      .required()
      .validate(x => {
        if (Set("local", "seq").contains(x.toLowerCase)) {
          success
        } else {
          failure("folderType only support local|seq")
        }
      })
    opt[String]("model")
      .text("BigDL model")
      .action((x, c) => c.copy(model = Some(x)))
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = Some(x)))
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = Some(x)))
    opt[Int]('b', "batch")
      .text("batch number")
      .action((x, c) => c.copy(batchPerPartition = x))
    opt[String]("classname")
      .text("file store class name")
      .action((x, c) => c.copy(classname = x))
      .required()
    opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
    opt[Boolean]('q', "quantize")
      .text("whether to quantize")
      .action((x, c) => c.copy(quantize = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PascolVocDemoParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("BigDL SSD Demo")
      val sc = new SparkContext(conf)
      Engine.init

      var model = if (params.model.isDefined) {
        // load BigDL model
        Module.loadModule(params.model.get)
      } else if (params.caffeDefPath.isDefined && params.caffeModelPath.isDefined) {
        // load caffe dynamically
        SSDCaffeLoader.loadCaffe(params.caffeDefPath.get, params.caffeModelPath.get)
      } else {
        throw new IllegalArgumentException(s"currently only support" +
          s" loading BigDL model or caffe model")
      }

      model = if (params.quantize) model.quantize() else model

      val data = params.folderType match {
        case "local" => ImageFrame.read(params.imageFolder, sc, params.nPartition)
        case "seq" =>
          val rdd = IOUtils.loadSeqFiles(params.nPartition, params.imageFolder, sc)
          ImageFrame.rdd(RecordToFeature()(rdd))
        case _ => throw new IllegalArgumentException(s"invalid folder name ${params.folderType}")
      }

      val labelMap = Source.fromFile(params.classname)
        .getLines().zipWithIndex.map(x => (x._2, x._1)).toMap
      val configure = Configure(
        ObjectDetectionConfig.preprocessSsdVgg(params.resolution, null, null),
        ScaleDetection(),
        batchPerPartition = params.batchPerPartition,
        labelMap)

      val predictor = new Predictor[Float](model, configure)

      val start = System.nanoTime()
      val output = predictor.predict(data)
      val recordsNum = output.toDistributed().rdd.count()
      val totalTime = (System.nanoTime() - start) / 1e9
      logger.info(s"[Prediction] $recordsNum in $totalTime seconds. Throughput is ${
        recordsNum / totalTime
      } record / sec")
      sc.stop()
    }
  }
}

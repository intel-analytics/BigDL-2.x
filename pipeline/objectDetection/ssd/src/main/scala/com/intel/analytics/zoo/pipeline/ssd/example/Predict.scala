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
import com.intel.analytics.bigdl.pipeline.ssd.IOUtils
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.zoo.pipeline.common.caffe.SSDCaffeLoader
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.Visualizer
import com.intel.analytics.zoo.pipeline.ssd._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.ssd.model.PreProcessParam
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
    outputFolder: String = "data/demo",
    folderType: String = "local",
    modelType: String = "vgg16",
    model: Option[String] = None,
    caffeDefPath: Option[String] = None,
    caffeModelPath: Option[String] = None,
    batch: Int = 8,
    savetxt: Boolean = true,
    vis: Boolean = true,
    classname: String = "",
    resolution: Int = 300,
    topK: Option[Int] = None,
    nPartition: Int = 1)

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
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]('t', "modelType")
      .text("net type : vgg16 | alexnet")
      .action((x, c) => c.copy(modelType = x))
      .required()
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
      .action((x, c) => c.copy(batch = x))
    opt[Boolean]('s', "savetxt")
      .text("whether to save detection results")
      .action((x, c) => c.copy(savetxt = x))
    opt[Boolean]('v', "vis")
      .text("whether to visualize the detections")
      .action((x, c) => c.copy(vis = x))
    opt[String]("classname")
      .text("file store class name")
      .action((x, c) => c.copy(classname = x))
      .required()
    opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
    opt[Int]('k', "topk")
      .text("return topk results")
      .action((x, c) => c.copy(topK = Some(x)))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PascolVocDemoParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("BigDL SSD Demo")
      val sc = new SparkContext(conf)
      Engine.init

      val classNames = Source.fromFile(params.classname).getLines().toArray

      val model = if (params.model.isDefined) {
        // load BigDL model
        Module.load[Float](params.model.get)
      } else if (params.caffeDefPath.isDefined && params.caffeModelPath.isDefined) {
        // load caffe dynamically
        SSDCaffeLoader.loadCaffe(params.caffeDefPath.get, params.caffeModelPath.get)
      } else {
        throw new IllegalArgumentException(s"currently only support" +
          s" loading BigDL model or caffe model")
      }

      val (data, paths) = params.folderType match {
        case "local" => IOUtils.loadLocalFolder(params.nPartition, params.imageFolder, sc)
        case "seq" => IOUtils.loadSeqFiles(params.nPartition, params.imageFolder, sc)
        case _ => throw new IllegalArgumentException(s"invalid folder name ${ params.folderType }")
      }

      val predictor = new SSDPredictor(model,
        PreProcessParam(params.batch, params.resolution,
          (123f, 117f, 104f), false, params.nPartition))

      val start = System.nanoTime()
      val output = predictor.predict(data)
      if (params.vis) output.cache()

      if (params.savetxt) {
        output.zip(paths).map { case (res: Tensor[Float], path: String) =>
          BboxUtil.resultToString(res, path)
        }.saveAsTextFile(params.outputFolder)
      } else {
        output.count()
      }

      val recordsNum = paths.count()
      val totalTime = (System.nanoTime() - start) / 1e9
      logger.info(s"[Prediction] ${ recordsNum } in $totalTime seconds. Throughput is ${
        recordsNum / totalTime
      } record / sec")

      if (params.vis) {
        if (params.folderType == "seq") {
          logger.warn("currently only support visualize local folder in Predict")
          return
        }

        paths.zip(output).foreach(pair => {
          val decoded = BboxUtil.decodeRois(pair._2)
          Visualizer.visDetection(pair._1, decoded, classNames, outPath = params.outputFolder)
        })
        logger.info(s"labeled images are saved to ${ params.outputFolder }")
      }
    }
  }
}

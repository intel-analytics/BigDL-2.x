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

package com.intel.analytics.zoo.pipeline.fasterrcnn.example

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.{PostProcessParam, PreProcessParam, PvanetFRcnn, VggFRcnn}
import com.intel.analytics.zoo.pipeline.fasterrcnn.Predictor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.common.{BboxUtil, IOUtils}
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.Visualizer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.io.Source

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.pipeline.fasterrcnn").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(imageFolder: String = "",
    outputFolder: String = "data/demo",
    folderType: String = "local",
    modelType: String = "vgg16",
    caffeDefPath: String = "",
    caffeModelPath: String = "",
    batch: Int = 8,
    visualize: Boolean = true,
    classname: String = "",
    nPartition: Int = 1)

  val predictParamParser = new OptionParser[PredictParam]("Spark-DL Demo") {
    head("Spark-DL Demo")
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
      .text("net type : vgg16 | alexnet | pvanet")
      .action((x, c) => c.copy(modelType = x))
      .required()
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = x))
      .required()
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = x))
      .required()
    opt[Int]('b', "batch")
      .text("batch number")
      .action((x, c) => c.copy(batch = x))
    opt[Boolean]('v', "visualize")
      .text("whether to visualize the detections")
      .action((x, c) => c.copy(visualize = x))
    opt[String]("classname")
      .text("file store class name")
      .action((x, c) => c.copy(classname = x))
      .required()
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    predictParamParser.parse(args, PredictParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("Spark-DL Faster RCNN Demo")
      val sc = new SparkContext(conf)
      Engine.init

      val classNames = Source.fromFile(params.classname).getLines().toArray
      val (model, preParam) = params.modelType match {
        case "vgg16" =>
          (Module.loadCaffe(VggFRcnn(classNames.length,
            PostProcessParam(0.3f, classNames.length, false, -1, 0)),
            params.caffeDefPath, params.caffeModelPath),
            PreProcessParam(params.batch, nPartition = params.nPartition))
        case "pvanet" =>
          (Module.loadCaffe(PvanetFRcnn(classNames.length,
            PostProcessParam(0.4f, classNames.length, true, -1, 0)),
            params.caffeDefPath, params.caffeModelPath),
            PreProcessParam(params.batch, Array(640), 32, nPartition = params.nPartition))
        case _ =>
          throw new Exception("unsupport network")
      }

      val (data, paths) = params.folderType match {
        case "local" => IOUtils.loadLocalFolder(params.nPartition, params.imageFolder, sc)
        case "seq" => IOUtils.loadSeqFiles(params.nPartition, params.imageFolder, sc)
        case _ => throw new IllegalArgumentException(s"invalid folder name ${ params.folderType }")
      }

      val predictor = new Predictor(model, preParam)

      val start = System.nanoTime()
      val output = predictor.predict(data)
      if (params.visualize) {
        output.cache()
      }
      val recordsNum = output.count()
      val totalTime = (System.nanoTime() - start) / 1e9
      logger.info(s"[Prediction] ${ recordsNum } in $totalTime seconds. Throughput is ${
        recordsNum / totalTime
      } record / sec")

      if (params.visualize) {
        paths.zip(output).foreach(pair => {
          val decoded = BboxUtil.decodeRois(pair._2)
          Visualizer.visDetection(pair._1, decoded, classNames, thresh = 0.6f,
            outPath = params.outputFolder)
        })

        logger.info(s"labeled images are saved to ${ params.outputFolder }")
      }
    }
  }
}

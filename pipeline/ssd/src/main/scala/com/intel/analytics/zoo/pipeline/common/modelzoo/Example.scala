/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.common.modelzoo


import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.common.{IOUtils, ModelType, ObjectDetect}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.pipeline.fasterrcnn").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(imageFolder: String = "",
    outputFolder: String = "data/demo",
    folderType: String = "local",
    modelType: String = "ssd",
    base: String = "vgg",
    bigdlModel: String = "",
    batch: Int = 8,
    visualize: Boolean = true,
    classname: String = "",
    resolution: Int = 300,
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
      .text("ssd | frcnn")
      .action((x, c) => c.copy(modelType = x))
      .required()
    opt[String]("base")
      .text("basenet: vgg | pvanet | mobilenet")
      .action((x, c) => c.copy(base = x))
      .required()
    opt[String]("model")
      .text("bigdl model path")
      .action((x, c) => c.copy(bigdlModel = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
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
    opt[Int]('r', "resolution")
      .text("input data resolution")
      .action((x, c) => c.copy(resolution = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    predictParamParser.parse(args, PredictParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("BigDL Object Detection Demo")
      val sc = new SparkContext(conf)
      Engine.init

      val classNames = IOUtils.loadClasses(params.classname)
      val model = Module.loadModule[Float](params.bigdlModel)

      val (data, paths) = params.folderType match {
        case "local" => IOUtils.loadLocalFolder(params.nPartition, params.imageFolder, sc)
        case "seq" => IOUtils.loadSeqFiles(params.nPartition, params.imageFolder, sc)
        case _ => throw new IllegalArgumentException(s"invalid folder name ${params.folderType}")
      }

      val input = params.modelType match {
        case ModelType.ssd =>
          params.base match {
            case ModelType.vgg =>
              IOUtils.preprocessSsdVgg(data, params.resolution, params.nPartition)
            case ModelType.mobilenet =>
              IOUtils.preprocessSsdMobilenet(data, params.resolution, params.nPartition)
          }
        case ModelType.frcnn =>
          params.base match {
            case ModelType.vgg =>
              IOUtils.preprocessFrcnnVgg(data, params.nPartition)
            case ModelType.pvanet =>
              IOUtils.preprocessFrcnnPvanet(data, params.nPartition)
          }
      }
      val output = ObjectDetect(input, model)

      if (params.visualize) {
        IOUtils.visualizeDetections(output, paths, classNames, params.outputFolder)
        logger.info(s"labeled images are saved to ${params.outputFolder}")
      } else {
        IOUtils.saveTextResults(output, paths, params.outputFolder)
      }
    }
  }
}


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

package com.intel.analytics.bigdl.pipeline.ssd.example

import com.intel.analytics.bigdl.dataset.image.Visualizer
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.pipeline.ssd._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.io.Source

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.pipeline.ssd").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PascolVocDemoParam(imageFolder: String = "",
    outputFolder: String = "data/demo",
    folderType: String = "local",
    modelPath: String = "",
    batch: Int = 8,
    vis: Boolean = true,
    classname: String = "")

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
    opt[String]("model")
      .text("BigDL model Path")
      .action((x, c) => c.copy(modelPath = x))
      .required()
    opt[Int]('b', "batch")
      .text("batch number")
      .action((x, c) => c.copy(batch = x))
    opt[Boolean]('v', "vis")
      .text("whether to visualize the detections")
      .action((x, c) => c.copy(vis = x))
    opt[String]("classname")
      .text("file store class name")
      .action((x, c) => c.copy(classname = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PascolVocDemoParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("Spark-DL SSD Demo")
      val sc = new SparkContext(conf)
      Engine.init
      val classNames = Source.fromFile(params.classname).getLines().toArray
      val model = Module.load[Float](params.modelPath)
      val nPartition = Engine.nodeNumber * Engine.coreNumber
      val data = params.folderType match {
        case "local" => IOUtils.loadLocalFolder(nPartition, params.imageFolder, sc)
        case "seq" => IOUtils.loadSeqFiles(nPartition, params.imageFolder, sc)
        case _ => throw new IllegalArgumentException(s"invalid folder name ${ params.folderType }")
      }

      val predictor = new Predictor(model,
        PreProcessParam(params.batch, resolution = 300, (123f, 117f, 104f), false))

      val start = System.nanoTime()
      val output = predictor.predict(data).collect()
      val recordsNum = output.length
      val totalTime = (System.nanoTime() - start) / 1e9
      logger.info(s"[Prediction] ${ recordsNum } in $totalTime seconds. Throughput is ${
        recordsNum / totalTime
      } record / sec")

      if (params.vis) {
        if (params.folderType == "seq") {
          logger.warn("currently only support visualize local folder in Predict")
          return
        }
        IOUtils.localImagePaths(params.imageFolder).zipWithIndex.foreach(pair => {
          var classIndex = 1
          val imgId = pair._2.toInt
          while (classIndex < classNames.length) {
            Visualizer.visDetection(pair._1, classNames(classIndex),
              output(imgId)(classIndex).classes,
              output(imgId)(classIndex).bboxes, thresh = 0.6f, outPath = params.outputFolder)
            classIndex += 1
          }
        })
        logger.info(s"labeled images are saved to ${ params.outputFolder }")
      }
    }
  }
}

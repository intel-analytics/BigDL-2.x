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

package com.intel.analytics.zoo.pipeline.common.model.example

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.pipeline.common.model.Model._
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.Visualizer
import com.intel.analytics.zoo.transform.vision.image.{ImageFrame}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

import scala.io.Source

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PascolVocDemoParam(imageFolder: String = "",
    outputFolder: String = "data/demo",
    source: String = "folder",
    model: String = "",
    savetxt: Boolean = true,
    vis: Boolean = true,
    classname: String = "",
    nPartition: Int = 1)

  val parser = new OptionParser[PascolVocDemoParam]("BigDL SSD Demo") {
    head("BigDL SSD Demo")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]("source")
      .text("data source type")
      .action((x, c) => c.copy(source = x))
      .required()
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]("model")
      .text("BigDL model")
      .action((x, c) => c.copy(model = x))
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

      val model = Module.loadModule[Float](params.model)

      val data = params.source match {
        case "folder" => ImageFrame.read(params.imageFolder, sc)
        case _ => throw new IllegalArgumentException(s"invalid folder name ${ params.source }")
      }

      val output = model.predictFeature(data)

      if (params.savetxt) {
        if (output.isDistributed()) {
          output.rdd.map(BboxUtil.detectionToString).saveAsTextFile(params.outputFolder)
        }
      }

      if (params.vis) {
        output.rdd.foreach(detection => {
          Visualizer.visualizeImageFeature(detection, classNames, outPath = params.outputFolder)
        })
        logger.info(s"labeled images are saved to ${ params.outputFolder }")
      }
    }
  }
}

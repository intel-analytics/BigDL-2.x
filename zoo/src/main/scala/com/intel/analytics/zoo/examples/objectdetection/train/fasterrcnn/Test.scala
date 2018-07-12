/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.examples.objectdetection.train.fasterrcnn

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.image.objectdetection.common.{IOUtils, MeanAveragePrecision}
import com.intel.analytics.zoo.models.image.objectdetection.fasterrcnn.{PostProcessParam, PreProcessParam}
import com.intel.analytics.zoo.pipeline.api.objectDetection.fasterrcnn.Validator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf}
import scopt.OptionParser

import scala.io.Source

object Test {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  case class TestParam(folder: String = "",
    modelType: String = "",
    imageSet: String = "voc_2007_test",
    model: String = "",
    className: String = "",
    batch: Int = 1,
    nPartition: Int = -1)

  val testParamParser = new OptionParser[TestParam]("Analyics Zoo fasterrcnn Test") {
    head("Analyics Zoo fasterrcnn Test")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
      .required()
    opt[String]('t', "modelType")
      .text("net type : vgg16 | pvanet")
      .action((x, c) => c.copy(modelType = x))
      .required()
    opt[String]('i', "imageset")
      .text("imageset: voc_2007_test")
      .action((x, c) => c.copy(imageSet = x))
      .required()
    opt[String]("model")
      .text("zoo model")
      .action((x, c) => c.copy(                   model = x))
    opt[String]("class")
      .text("class file")
      .action((x, c) => c.copy(className = x))
      .required()
    opt[Int]('b', "batch")
      .text("batch number")
      .action((x, c) => c.copy(batch = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
  }

  def main(args: Array[String]) {
    testParamParser.parse(args, TestParam()).foreach { params =>
      val conf = new SparkConf().setAppName(s"Analytics Zoo Faster-RCNN Test ${params.model}")
      val sc = NNContext.initNNContext(conf)

      val classes = Source.fromFile(params.className).getLines().toArray
      val evaluator = new MeanAveragePrecision(true, normalized = false,
        classes = classes)
      val rdd = IOUtils.loadSeqFiles(params.nPartition, params.folder, sc)

      val model = Module.loadModule[Float](params.model)

      val (preParam, postParam) = params.modelType.toLowerCase() match {
        case "vgg16" =>
          val postParam = PostProcessParam(0.3f, classes.length, false, 100, 0.05)
          val preParam = PreProcessParam(params.batch, nPartition = params.nPartition)
          (preParam, postParam)
        case "pvanet" =>
          val postParam = PostProcessParam(0.4f, classes.length, true, 100, 0.05)
          val preParam = PreProcessParam(params.batch, Array(640), 32,
            nPartition = params.nPartition)
          (preParam, postParam)
        case _ =>
          throw new Exception("unsupport network")
      }
      val validator = new Validator(model, preParam, postParam, evaluator)
      validator.test(rdd)
      sc.stop()
    }
  }
}

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

import com.intel.analytics.bigdl.nn.{Graph, Module}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.common.{IOUtils, MeanAveragePrecision}
import com.intel.analytics.zoo.pipeline.fasterrcnn.Validator
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.{PostProcessParam, PreProcessParam, VggFRcnn}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

object Test {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.pipeline.fasterrcnn").setLevel(Level.INFO)

  case class TestParam(folder: String = "",
    modelType: String = "",
    imageSet: String = "voc_2007_test",
    bigdlModel: String = "",
    caffeDefPath: String = "",
    caffeModelPath: String = "",
    nClass: Int = 21,
    batch: Int = 1,
    nPartition: Int = -1,
    isProtobuf: Boolean = true)

  val testParamParser = new OptionParser[TestParam]("Spark-DL Test") {
    head("Spark-DL Test")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
      .required()
    opt[String]('t', "modelType")
      .text("net type : vgg16 | alexnet | pvanet")
      .action((x, c) => c.copy(modelType = x))
      .required()
    opt[String]('i', "imageset")
      .text("imageset: voc_2007_test")
      .action((x, c) => c.copy(imageSet = x))
      .required()
    opt[String]("model")
      .text("bigdl model")
      .action((x, c) => c.copy(bigdlModel = x))
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = x))
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = x))
    opt[Int]("nclass")
      .text("class number")
      .action((x, c) => c.copy(nClass = x))
      .required()
    opt[Int]('b', "batch")
      .text("batch number")
      .action((x, c) => c.copy(batch = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
    opt[Boolean]("protobuf")
      .text("is model saved with protobuf")
      .action((x, c) => c.copy(isProtobuf = x))
  }

  def main(args: Array[String]) {
    testParamParser.parse(args, TestParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("Spark-DL Faster RCNN Test")
      val sc = new SparkContext(conf)
      Engine.init

      val evaluator = new MeanAveragePrecision(true, normalized = false,
        nClass = params.nClass)
      val rdd = IOUtils.loadSeqFiles(params.nPartition, params.folder, sc)

      val model = if (params.isProtobuf) Module.loadCaffe(VggFRcnn(params.nClass,
        PostProcessParam(0.3f, params.nClass, false, 100, 0.05)),
        params.caffeDefPath, params.caffeModelPath)
      else Module.load[Float](params.bigdlModel)

      val (preParam, postParam) = params.modelType.toLowerCase() match {
        case "vgg16" =>
          val postParam = PostProcessParam(0.3f, params.nClass, false, 100, 0.05)
          val preParam = PreProcessParam(params.batch, nPartition = params.nPartition)
          (preParam, postParam)
        case "pvanet" =>
          val postParam = PostProcessParam(0.4f, params.nClass, true, 100, 0.05)
          val preParam = PreProcessParam(params.batch, Array(640), 32,
            nPartition = params.nPartition)
          (preParam, postParam)
        case _ =>
          throw new Exception("unsupport network")
      }
      val validator = new Validator(model, preParam, postParam, evaluator)
      validator.test(rdd._1)
      sc.stop()
    }
  }
}

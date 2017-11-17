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
import com.intel.analytics.zoo.pipeline.common.MeanAveragePrecision
import com.intel.analytics.zoo.pipeline.ssd._
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.common.caffe.SSDCaffeLoader
import com.intel.analytics.zoo.pipeline.ssd.model.PreProcessParam
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

object Test {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.models.fasterrcnn").setLevel(Level.INFO)

  case class PascolVocTestParam(folder: String = "",
    modelType: String = "vgg16",
    imageSet: String = "voc_2007_test",
    model: Option[String] = None,
    caffeDefPath: Option[String] = None,
    caffeModelPath: Option[String] = None,
    batch: Int = 8,
    nClass: Int = 0,
    resolution: Int = 300,
    useNormalized: Boolean = false,
    nPartition: Int = 1)

  val parser = new OptionParser[PascolVocTestParam]("BigDL SSD Test") {
    head("BigDL SSD Test")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
      .required()
    opt[String]('t', "modelType")
      .text("net type : VGG16 | PVANET")
      .action((x, c) => c.copy(modelType = x))
      .required()
    opt[String]('i', "imageset")
      .text("imageset: voc_2007_test")
      .action((x, c) => c.copy(imageSet = x))
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
    opt[Int]("nclass")
      .text("class number")
      .action((x, c) => c.copy(nClass = x))
      .required()
    opt[Int]('r', "resolution")
      .text("input resolution 300 or 512")
      .action((x, c) => c.copy(resolution = x))
      .required()
    opt[Boolean]("normalize")
      .text("whether to use normalized detection")
      .action((x, c) => c.copy(useNormalized = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
  }

  def main(args: Array[String]) {
    parser.parse(args, PascolVocTestParam()).foreach { params =>
      val conf = Engine.createSparkConf().setAppName("BigDL SSD Test")
      val sc = new SparkContext(conf)
      Engine.init

      val evaluator = new MeanAveragePrecision(true, normalized = params.useNormalized,
        nClass = params.nClass)
      val rdd = IOUtils.loadSeqFiles(params.nPartition, params.folder, sc)._1

      val model = if (params.model.isDefined) {
        // load BigDL model
        Module.load[Float](params.model.get)
      } else if (params.caffeDefPath.isDefined && params.caffeModelPath.isDefined) {
        // load caffe dynamically
        SSDCaffeLoader.loadCaffe(params.caffeDefPath.get, params.caffeModelPath.get)
      } else {
        throw new IllegalArgumentException(s"currently only support loading BigDL model or caffe model")
      }

      val validator = new Validator(model, PreProcessParam(params.batch, params.resolution,
        (123f, 117f, 104f), true, params.nPartition), evaluator, useNormalized = params.useNormalized)

      validator.test(rdd)
    }
  }
}

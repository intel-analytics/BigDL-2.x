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
package com.intel.analytics.zoo.pipeline.common

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.zoo.pipeline.common.caffe.{FrcnnCaffeLoader, SSDCaffeLoader}
import scopt.OptionParser

object WeightConverter {

  case class WeightParams(
    model: String = ".",
    output: String = "."
  )

  private val parser = new OptionParser[WeightParams]("Weight Converter") {
    head("Weight Converter")
    opt[String]('m', "model")
      .text("model path")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[String]('o', "output path")
      .text("where you put the generated weight file")
      .action((x, c) => c.copy(output = x))
      .required()
  }


  def main(args: Array[String]): Unit = {
    parser.parse(args, WeightParams()).foreach(param => {
      val model = Module.load[Float](param.model)
      model.saveWeights(param.output, true)
    })
  }
}

object CaffeConverter {

  case class CaffeConverterParam(
    caffeDefPath: String = "",
    caffeModelPath: String = "",
    bigDLModel: String = "",
    modelType: String = "")

  val parser = new OptionParser[CaffeConverterParam]("BigDL SSD Caffe Converter") {
    head("BigDL SSD Caffe Converter")
    opt[String]("caffeDefPath")
      .text("caffe prototxt")
      .action((x, c) => c.copy(caffeDefPath = x))
      .required()
    opt[String]("caffeModelPath")
      .text("caffe model path")
      .action((x, c) => c.copy(caffeModelPath = x))
      .required()
    opt[String]('o', "output")
      .text("output bigDL model")
      .action((x, c) => c.copy(bigDLModel = x))
    opt[String]('t', "modelType")
      .text("model type: general, ssd or frcnn")
      .action((x, c) => {
        c.copy(modelType = x)
      })
  }

  def main(args: Array[String]) {
    parser.parse(args, CaffeConverterParam()).foreach { params =>
      val model = params.modelType match {
        case "ssd" => SSDCaffeLoader.loadCaffe(params.caffeDefPath, params.caffeModelPath)
        case "frcnn" => FrcnnCaffeLoader.loadCaffe(params.caffeDefPath, params.caffeModelPath)
        case _ => Module.loadCaffeModel[Float](params.caffeDefPath, params.caffeModelPath)
      }
      model.saveModule(params.bigDLModel, overWrite = true)
    }
  }
}

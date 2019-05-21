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

package com.intel.analytics.zoo.examples.vnni.openvino

import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

case class CalibrationParams(model: String = "",
                             outputDir: String = ".",
                             modelType: String = "resnet_v1_50",
                             inputShape: Array[Int] = Array(4, 224, 224, 3),
                             ifReverseInputChannels: Boolean = true,
                             meanValues: Array[Float] = Array(123.68f,
                               116.78f, 103.94f),
                             inputScale: Float = 1f,
                             networkType: String = "C",
                             validationFilePath: String = "",
                             subset: Int = 32,
                             openCVLibs: String = "/tmp")

object PrepareOpenVINOModel {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[CalibrationParams]("ResNet50 Int8 Inference Example") {
      opt[String]('m', "model")
        .text("The path to TensorFlow model")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[String]('o', "outputDir")
        .text("Output path of prepared model")
        .action((x, c) => c.copy(outputDir = x))
      opt[String]("modelType")
        .text("Model type of this model, e.g., resnet_v1_50")
        .action((x, c) => c.copy(modelType = x))
      opt[Array[Int]]("inputShape")
        .text("Input shape of this model, e.g., [4, 224, 224, 3]")
        .action((x, c) => c.copy(inputShape = x))
      opt[Boolean]("ifReverseInputChannels")
        .text("Reverse Input Channels or not")
        .action((x, c) => c.copy(ifReverseInputChannels = x))
      opt[Array[Float]]("meanValues")
        .text("Model type of this model, e.g., ")
        .action((x, c) => c.copy(meanValues = x))
      opt[Float]("inputScale")
        .text("Model type of this model, e.g., ")
        .action((x, c) => c.copy(inputScale = x))
      opt[String]("networkType")
        .text("Model type of this model, e.g., ")
        .action((x, c) => c.copy(networkType = x))
      opt[String]("validationFilePath")
        .text("Model type of this model, e.g., ")
        .action((x, c) => c.copy(validationFilePath = x))
      opt[Int]("subset")
        .text("Model type of this model, e.g., ")
        .action((x, c) => c.copy(subset = x))
      opt[String]("openCVLibs")
        .text("Model type of this model, e.g., ")
        .action((x, c) => c.copy(openCVLibs = x))
    }
    parser.parse(args, CalibrationParams()).foreach(param => {
      InferenceModel.doCalibrateTF(modelPath = param.model,
        networkType = param.networkType,
        validationFilePath = param.validationFilePath,
        subset = param.subset,
        opencvLibPath = param.openCVLibs,
        outputDir = param.outputDir
      )
    })
  }
}

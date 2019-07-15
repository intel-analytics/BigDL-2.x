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

case class PrepareOpenVINOResNetParams(model: String = "",
                                       batchSize: Int = 4,
                                       validationFilePath: String = "",
                                       subset: Int = 32,
                                       openCVLibs: String = "/tmp")

object PrepareOpenVINOResNet {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[PrepareOpenVINOResNetParams]("Prepare ResNet50 Int8 Model") {
      opt[String]('m', "model")
        .text("Dir that contains to TensorFlow model checkpoint")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[String]('v', "validationFilePath")
        .text("Dir of Validation images and val.txt")
        .action((x, c) => c.copy(validationFilePath = x))
        .required()
      opt[Int]("subset")
        .text("Number of images in val.txt, default 32")
        .action((x, c) => c.copy(subset = x))
      opt[String]('l', "openCVLibs")
        .text("Dir of downloaded OpenCV libs")
        .action((x, c) => c.copy(openCVLibs = x))
        .required()
      opt[Int]('b', "batchSize")
        .text("Input batch size")
        .action((x, c) => c.copy(batchSize = x))
    }
    parser.parse(args, PrepareOpenVINOResNetParams()).foreach(param => {
      InferenceModel.doOptimizeTF(modelPath = null,
        imageClassificationModelType = "resnet_v1_50",
        checkpointPath = param.model + "/resnet_v1_50.ckpt",
        inputShape = Array(param.batchSize, 224, 224, 3),
        ifReverseInputChannels = true,
        meanValues = Array(123.68f, 116.78f, 103.94f),
        scale = 1f,
        outputDir = param.model
      )
      InferenceModel.doCalibrateTF(modelPath = param.model + "/resnet_v1_50_inference_graph.xml",
        networkType = "C",
        validationFilePath = param.validationFilePath + "/val.txt",
        subset = param.subset,
        opencvLibPath = param.openCVLibs,
        outputDir = param.model
      )
    })
  }
}

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

package com.intel.analytics.zoo.examples.vnni.bigdl

import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.imageclassification.ImageClassifier
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser


case class ImageNetInferenceParams(folder: String = "./",
                                   model: String = "",
                                   batchSize: Int = 128)

object ImageNetInference {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.engineType", "mkldnn")
    val parser = new OptionParser[ImageNetInferenceParams]("ImageNet Int8 Example") {
      opt[String]('f', "folder")
        .text("The path to the imagenet dataset for inference")
        .action((x, c) => c.copy(folder = x))
        .required()
      opt[String]('m', "model")
        .text("The path to the int8 quantized ResNet50 model snapshot")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[Int]('b', "batchSize")
        .text("batch size")
        .action((x, c) => c.copy(batchSize = x))
    }
    parser.parse(args, ImageNetInferenceParams()).map(param => {
      val sc = NNContext.initNNContext("ImageNet2012 with Int8 Inference Example")
      val images = ImageSet.readSeqFiles(param.folder, sc)
      val model = ImageClassifier.loadModel[Float](param.model, quantize = true)
      val result = model.evaluateImageSet(images, param.batchSize,
        Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))

      result.foreach(r => println(s"${r._2} is ${r._1}"))

      sc.stop()
    })
  }
}

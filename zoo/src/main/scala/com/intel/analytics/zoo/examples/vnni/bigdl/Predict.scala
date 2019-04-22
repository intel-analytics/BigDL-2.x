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

package com.intel.analytics.zoo.examples.vnni.bigdl

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.imageclassification.{ImageClassifier, LabelOutput}
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

case class ImageClassificationParams(folder: String = "./",
                                     model: String = "",
                                     topN: Int = 5,
                                     partitionNum: Int = 4)

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.engineType", "mkldnn")
    val parser = new OptionParser[ImageClassificationParams]("ResNet50 Int8 Inference Example") {
      opt[String]('f', "folder")
        .text("The path to the image data")
        .action((x, c) => c.copy(folder = x))
        .required()
      opt[String]('m', "model")
        .text("The path to the int8 quantized ResNet50 model snapshot")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[Int]("topN")
        .text("top N number")
        .action((x, c) => c.copy(topN = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
    }
    parser.parse(args, ImageClassificationParams()).map(param => {
      val sc = NNContext.initNNContext("ResNet50 Int8 Inference Example")
      val images = ImageSet.read(param.folder)
      val model = ImageClassifier.loadModel[Float](param.model, quantize = true)
      val output = model.predictImageSet(images)
      val labelOutput = LabelOutput(model.getConfig().labelMap, "clses",
        "probs", probAsInput = false)
      val result = labelOutput(output).toLocal().array

      logger.info(s"Prediction result")
      result.foreach(imageFeature => {
        logger.info(s"image : ${imageFeature.uri}, top ${param.topN}")
        val clses = imageFeature("clses").asInstanceOf[Array[String]]
        val probs = imageFeature("probs").asInstanceOf[Array[Float]]
        for (i <- 0 until param.topN) {
          logger.info(s"\t class : ${clses(i)}, credit : ${probs(i)}")
        }
      })

      sc.stop()
    })
  }
}

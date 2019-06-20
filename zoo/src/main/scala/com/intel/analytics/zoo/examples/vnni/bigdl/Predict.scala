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
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser

case class PredictParams(folder: String = "./",
                         model: String = "",
                         topN: Int = 5)

object Predict {
  Logger.getLogger("com.intel.analytics.bigdl.transform.vision").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.engineType", "mkldnn")

    val parser = new OptionParser[PredictParams]("ResNet50 Int8 Inference Example") {
      opt[String]('f', "folder")
        .text("The local folder path that contains images for prediction")
        .action((x, c) => c.copy(folder = x))
        .required()
      opt[String]('m', "model")
        .text("The path to the downloaded int8 model snapshot")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[Int]("topN")
        .text("The top N classes with highest probabilities as output")
        .action((x, c) => c.copy(topN = x))
    }
    parser.parse(args, PredictParams()).map(param => {
      val sc = NNContext.initNNContext(
        "Image classification inference example using int8 quantized model")
      val images = ImageSet.read(param.folder, imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)
      val model = ImageClassifier.loadModel[Float](param.model)
      logger.info(s"Start inference on images under ${param.folder}...")
      val output = model.predictImageSet(images)
      val labelOutput = LabelOutput(model.getConfig().labelMap, probAsOutput = false)
      val results = labelOutput(output).toLocal().array

      logger.info(s"Prediction results:")
      results.foreach(imageFeature => {
        logger.info(s"image: ${imageFeature.uri}, top ${param.topN}")
        val classes = imageFeature("classes").asInstanceOf[Array[String]]
        val probs = imageFeature("probs").asInstanceOf[Array[Float]]
        for (i <- 0 until param.topN) {
          logger.info(s"\t class: ${classes(i)}, credit: ${probs(i)}")
        }
      })
      logger.info(s"Prediction finished.")
      sc.stop()
    })
  }
}

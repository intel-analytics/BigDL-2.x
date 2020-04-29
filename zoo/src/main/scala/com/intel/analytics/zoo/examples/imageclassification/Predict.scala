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

package com.intel.analytics.zoo.examples.imageclassification

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.imageclassification.{ImageClassifier, LabelOutput}
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser


object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)

  val logger = Logger.getLogger(getClass)

  case class TopNClassificationParam(imageFolder: String = "",
                                     model: String = "",
                                     topN: Int = 5,
                                     nPartition: Int = 1)

  val parser = new OptionParser[TopNClassificationParam]("ImageClassification demo") {
    head("Analytics Zoo ImageClassification demo")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]("model")
      .text("Analytics Zoo model")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[Int]("topN")
      .text("top N number")
      .action((x, c) => c.copy(topN = x))
      .required()
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, TopNClassificationParam()).foreach { params =>
      val sc = NNContext.initNNContext("Image Classification")
      val model = ImageClassifier.loadModel[Float](params.model)
      val data = ImageSet.read(params.imageFolder, sc, params.nPartition)
      val output = model.predictImageSet(data)
      val labelOutput = LabelOutput(model.getConfig.labelMap, "clses", "probs")
      val result = labelOutput(output).toDistributed().rdd.collect

      logger.info(s"Prediction result")
      result.foreach(imageFeature => {
        logger.info(s"image : ${imageFeature.uri}, top ${params.topN}")
        val clsses = imageFeature("clses").asInstanceOf[Array[String]]
        val probs = imageFeature("probs").asInstanceOf[Array[Float]]
        for (i <- 0 until params.topN) {
          logger.info(s"\t class : ${clsses(i)}, credit : ${probs(i)}")
        }
      })
      println("finished...")
      sc.stop()
    }
  }
}

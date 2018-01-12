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

package com.intel.analytics.zoo.models.imageclassification.example

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.transform.vision.image.ImageFrame
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.zoo.models.Predictor
import com.intel.analytics.zoo.models.imageclassification.util.{Consts, LabelOutput}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser


object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class TopNClassificationParam(imageFolder: String = "",
                                     model: String = "",
                                     topN: Int = 5,
                                     nPartition: Int = 1)

  val parser = new OptionParser[TopNClassificationParam]("ImageClassification demo") {
    head("BigDL ImageClassification demo")
    opt[String]('f', "folder")
      .text("where you put the demo image data")
      .action((x, c) => c.copy(imageFolder = x))
      .required()
    opt[String]("model")
      .text("BigDL model")
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
      val conf = Engine.createSparkConf().setAppName("BigDL ImageClassification Demo")
        .setMaster("local[1]")
      val sc = new SparkContext(conf)
      Engine.init
      val model = Module.loadModule[Float](params.model)
      val data = ImageFrame.read(params.imageFolder, sc)
      val predictor = Predictor(model)
      val labelOutput = LabelOutput(predictor.configure.labelMap, "clses", "probs")
      val predict = predictor.predict(data)

      val result = labelOutput(predict).toDistributed().rdd.collect

      logger.info(s"Prediction result")
      result.foreach(imageFeature => {
        logger.info(s"image : ${imageFeature.uri}, top ${params.topN}")
        val clsses = imageFeature("clses").asInstanceOf[Array[String]]
        val probs = imageFeature("probs").asInstanceOf[Array[Float]]
        for (i <- 0 until params.topN) {
          logger.info(s"\t class : ${clsses(i)}, credit : ${probs(i)}")
        }
      })
    }
  }
}

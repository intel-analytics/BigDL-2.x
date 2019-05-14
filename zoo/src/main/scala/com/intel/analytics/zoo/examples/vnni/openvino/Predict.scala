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

import com.intel.analytics.bigdl.dataset.SampleToMiniBatch
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.{ImageCenterCrop, ImageMatToTensor, ImageResize, ImageSet, ImageSetToSample}
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

case class ImageClassificationParams(folder: String = "./",
                                     model: String = "",
                                     weight: String = "",
                                     batchSize: Int = 4,
                                     topN: Int = 5,
                                     partitionNum: Int = 4)

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[ImageClassificationParams]("ResNet50 Int8 Inference Example") {
      opt[String]('f', "folder")
        .text("The path to the image data")
        .action((x, c) => c.copy(folder = x))
        .required()
      opt[String]('m', "model")
        .text("The path to the int8 ResNet50 model")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[String]('w', "weight")
        .text("The path to the int8 ResNet50 model weight")
        .action((x, c) => c.copy(weight = x))
      opt[Int]("topN")
        .text("top N number")
        .action((x, c) => c.copy(topN = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]('b', "batchSize")
        .text("batch size")
        .action((x, c) => c.copy(batchSize = x))
    }
    parser.parse(args, ImageClassificationParams()).map(param => {
      val sc = NNContext.initNNContext("ImageNet2012 with OpenVINO Int8 Inference Example")

      val model = new InferenceModel(1)
      model.doLoadOpenVINOInt8(param.model, param.weight, param.batchSize)

      // Read ImageNet val
      val images = ImageSet.read(param.folder, sc).toDistributed()
      // If the actual partitionNum of sequence files is too large, then the
      // total batchSize we calculate (partitionNum * batchPerPartition) would be
      // too large for inference.
      // mkldnn runs a single model and single partition on a single node.
      if (images.rdd.partitions.length > param.partitionNum) {
        images.rdd = images.rdd.coalesce(param.partitionNum, shuffle = false)
      }
      // Pre-processing
      val inputs = images ->
        ImageResize(256, 256) ->
        ImageCenterCrop(224, 224) ->
        ImageMatToTensor() ->
        ImageSetToSample()
      val batched = inputs.toDataSet() -> SampleToMiniBatch(param.batchSize)

      // Predict
      logger.debug("Begin Prediction")
      val start = System.nanoTime()
      val results = batched.toDistributed().data(false).map { miniBatch =>
        model.doPredictInt8(miniBatch.getInput.toTensor.addSingletonDimension())
      }
      val timeUsed = System.nanoTime() - start
      // Post-processing
      val throughput = "%.2f".format(images.toDistributed().rdd.count()
        .toFloat / (timeUsed / 1e9))
      val batchLatency = "%.2f".format(timeUsed / 1e6 / results.count().toFloat)
      logger.info(s"Takes $timeUsed ns, throughput is $throughput imgs/sec")
      logger.info(s"Average Predict latency is $batchLatency ms")
      // Evaluation
      // Compare labels and output results
      sc.stop()
    })
  }
}
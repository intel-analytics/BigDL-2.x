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
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy}
import com.intel.analytics.zoo.feature.image.{ImageCenterCrop, ImageMatToTensor, ImageResize, ImageSet, ImageSetToSample}
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser


case class ImageNetEvaluationParams(folder: String = "./",
                                    model: String = "",
                                    weight: String = "",
                                    batchSize: Int = 4,
                                    partitionNum: Int = 32,
                                    isInt8: Boolean = false)

object ImageNetEvaluation {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[ImageNetEvaluationParams]("ImageNet Int8 Example") {
      opt[String]('f', "folder")
        .text("The path to the imagenet dataset for inference")
        .action((x, c) => c.copy(folder = x))
        .required()
      opt[String]('m', "model")
        .text("The path to the int8 quantized ResNet50 model snapshot")
        .action((x, c) => c.copy(model = x))
        .required()
      opt[String]('w', "weight")
        .text("The path to the int8 ResNet50 model weight")
        .action((x, c) => c.copy(weight = x))
      opt[Int]('b', "batchSize")
        .text("batch size")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]("partitionNum")
        .text("The partition number of the dataset")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Boolean]("isInt8")
        .text("Is Int8 optimized model?")
        .action((x, c) => c.copy(isInt8 = x))
    }
    parser.parse(args, ImageNetEvaluationParams()).foreach(param => {
      val sc = NNContext.initNNContext("ImageNet2012 with OpenVINO Evaluation Example")


      val model = new InferenceModel(1)
      if (param.isInt8) {
        model.doLoadOpenVINOInt8(param.model, param.weight, param.batchSize)
      } else {
        model.doLoadOpenVINO(param.model, param.weight)
      }

      // Read ImageNet val
      val images = ImageSet.readSequenceFiles(param.folder, sc, param.partitionNum)
      val vMethods = Array(new Top1Accuracy[Float], new Top5Accuracy[Float])
      val validations = images.rdd.sparkContext.broadcast(vMethods)
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

      val imageNum = images.toDistributed().rdd.count()
        .toFloat

      // Predict
      logger.debug("Begin Prediction")
      val start = System.nanoTime()
      val results = batched.toDistributed().data(false).map {miniBatch =>
        val predict = model.doPredict(miniBatch
          .getInput.toTensor.addSingletonDimension())
        val localMethod = validations.value
        localMethod.map(valMethod => {
          valMethod(predict.toTensor.apply(1), miniBatch.getTarget())
        })
      }.reduce((left, right) => {
        left.zip(right).map { case (l, r) => l + r }
      }).zip(vMethods)
      println("Evaluation Results:")
      results.foreach(r => println(s"${r._2} is ${r._1}"))
      val timeUsed = System.nanoTime() - start
      // Post-processing
      val throughput = "%.2f".format(imageNum / (timeUsed / 1e9))
      logger.info(s"Takes $timeUsed ns, throughput is $throughput FPS (imgs/sec)")
      // Evaluation
      // Compare labels and output results
      sc.stop()
    })
  }
}

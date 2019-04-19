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

package com.intel.analytics.zoo.examples.vnni.openvino

import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.{ImageBytesToMat, ImageMatToFloats, ImageSet}
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.log4j.{Level, Logger}
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser


case class ImageNetInferenceParams(folder: String = "./",
                                   model: String = "",
                                   weight: String = "",
                                   batchSize: Int = 4)

object ImageNetInference {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[ImageNetInferenceParams]("ImageNet Int8 Example") {
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
        .action((x, c) => c.copy(model = x))
      opt[Int]('b', "batchSize")
        .text("batch size")
        .action((x, c) => c.copy(batchSize = x))
    }
    parser.parse(args, ImageNetInferenceParams()).map(param => {
      val sc = NNContext.initNNContext("ImageNet2012 with Int8 Inference Example")
      val images = ImageSet.read(param.folder, sc)
      val model = new InferenceModel(4)
      model.doLoadOpenVINO(param.model, param.weight)

      images -> ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)
      images -> ImageMatToFloats(validHeight = 224, validWidth = 224)
      val output = images.toDistributed().rdd.foreach { img =>
        model.doPredict(img.toTensor(ImageFeature.floats))
      }
      // Evaluation
      // Read val.txt
      // Compare labels and output results
      sc.stop()
    })
  }
}
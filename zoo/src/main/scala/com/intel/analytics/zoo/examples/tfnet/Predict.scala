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

package com.intel.analytics.zoo.examples.tfnet

import java.nio.file.Paths

import com.intel.analytics.bigdl.nn.{Contiguous, Sequential, Transpose}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.{ImageMatToTensor, ImageResize, ImageSet, ImageSetToSample}
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser


object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(
    image: String = "/tmp/datasets/cat_dog/train_sampled",
    outputFolder: String = "data/demo",
    model: String = "/tmp/models/ssd_mobilenet_v1_coco_2017_11_17" +
      "/frozen_inference_graph.pb",
    classNamePath: String = "/tmp/models/coco_classname.txt",
    nPartition: Int = 4)

  val parser = new OptionParser[PredictParam]("TFNet Object Detection Example") {
    head("TFNet Object Detection Example")
    opt[String]('i', "image")
      .text("where you put the demo image data, can be image folder or image path")
      .action((x, c) => c.copy(image = x))
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
    opt[String]('c', "classNamePath")
      .text("where you put the class name file")
      .action((x, c) => c.copy(outputFolder = x))
    opt[String]("model")
      .text("BigDL model")
      .action((x, c) => c.copy(model = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PredictParam()).foreach { params =>

      val sc = NNContext.getNNContext("TFNet Object Detection Example")

      val inputs = Seq("ToFloat:0")
      val outputs = Seq("num_detections:0", "detection_boxes:0",
        "detection_scores:0", "detection_classes:0")

      val detector = TFNet(params.model, inputs, outputs)
      val model = Sequential()
      model.add(Transpose(Array((2, 4), (2, 3))))
      model.add(Contiguous())
      model.add(detector)

      val data = ImageSet.read(params.image, sc, minPartitions = params.nPartition)
        .transform(ImageResize(256, 256) -> ImageMatToTensor() -> ImageSetToSample())
      val output = model.predictImage(data.toImageFrame(), batchPerPartition = 1)

      // print the first result
      val result = output.toDistributed().rdd.first().predict()
      println(result)
    }
  }

  private def getOutPath(outPath: String, uri: String, encoding: String): String = {
    Paths.get(outPath,
      s"detection_${ uri.substring(uri.lastIndexOf("/") + 1,
        uri.lastIndexOf(".")) }.${encoding}").toString
  }
}

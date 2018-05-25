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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.feature.image.{ImageMatToTensor, ImageResize, ImageSet, ImageSetToSample}
import com.intel.analytics.zoo.models.image.objectdetection.{ScaleDetection, Visualizer}
import com.intel.analytics.zoo.pipeline.api.net.TFNet
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import scopt.OptionParser

import scala.io.Source

object Predict {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(
    image: String = "/tmp/datasets/cat_dog/train_sampled",
    outputFolder: String = "data/demo",
    model: String = "/home/yang/applications/ssd_mobilenet_v1_coco_2017_11_17" +
      "/frozen_inference_graph.pb",
    classNamePath: String = "./coco_classname.txt",
    nPartition: Int = 4)

  val parser = new OptionParser[PredictParam]("Analytics Zoo Object Detection Demo") {
    head("Analytics Zoo Object Detection Demo")
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
      val conf = new SparkConf()
        .setAppName("Object Detection Example")
      val sc = NNContext.getNNContext(conf)

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

      val prediction = output.transform(new FeatureTransformer {
        override protected def transformMat(feature: ImageFeature): Unit = {
          // only pick the first detection
          val output = feature.predict().asInstanceOf[Table]
          val numDetections = output[Tensor[Float]](1).valueAt(1).toInt
          val boxes = output[Tensor[Float]](2)
          val (ymin, xmin, ymax, xmax) =
            (boxes.valueAt(1, 1, 1), boxes.valueAt(1, 1, 2),
              boxes.valueAt(1, 1, 3), boxes.valueAt(1, 1, 4))
          val score = output[Tensor[Float]](3).valueAt(1, 1)
          val clas = output[Tensor[Float]](4).valueAt(1, 1)
          val pred = Tensor[Float](Array(clas, score, xmin, ymin, xmax, ymax), Array(1, 6))
          feature.update(ImageFeature.predict, pred)
        }
      } -> ScaleDetection())

      val labelMap = Source.fromFile(params.classNamePath)
        .getLines().zipWithIndex.map(x => (x._2, x._1)).toMap

      val visualizer = Visualizer(labelMap, encoding = "jpg")
      val visualized = visualizer(prediction).toDistributed()
      val result = visualized.rdd.map(imageFeature =>
        (imageFeature.uri(), imageFeature[Array[Byte]](Visualizer.visualized))).collect()

      result.foreach(x => {
        Utils.saveBytes(x._2, getOutPath(params.outputFolder, x._1, "jpg"), true)
      })
      logger.info(s"labeled images are saved to ${params.outputFolder}")
    }
  }

  private def getOutPath(outPath: String, uri: String, encoding: String): String = {
    Paths.get(outPath,
      s"detection_${ uri.substring(uri.lastIndexOf("/") + 1,
        uri.lastIndexOf(".")) }.${encoding}").toString
  }
}

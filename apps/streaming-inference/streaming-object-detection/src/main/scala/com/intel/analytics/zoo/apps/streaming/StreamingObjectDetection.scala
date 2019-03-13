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

package com.intel.analytics.zoo.apps.streaming

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.feature.image.{ImageBytesToMat, ImageSet}
import com.intel.analytics.zoo.models.image.objectdetection.{ObjectDetector, Visualizer}
import org.apache.commons.io.FileUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser

object StreamingObjectDetection {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(image: String = "file:///tmp/zoo/streaming",
                          outputFolder: String = "data/demo",
                          modelPath: String = "",
                          nPartition: Int = 1)

  val parser = new OptionParser[PredictParam]("Analytics Zoo Object Detection Demo") {
    head("Analytics Zoo Object Detection Demo")
    opt[String]('i', "image")
      .text("where you put the demo image data, can be image folder or image path")
      .action((x, c) => c.copy(image = x))
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]("modelPath")
      .text("Analytics Zoo model path")
      .action((x, c) => c.copy(modelPath = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PredictParam()).foreach { params =>
      val sc = NNContext.initNNContext("Streaming Object Detection")
      val ssc = new StreamingContext(sc, Seconds(3))

      // Load pre-trained model
      val model = ObjectDetector.loadModel[Float](params.modelPath)

      val lines = ssc.textFileStream(params.image)
      lines.foreachRDD { batchPath =>
        // Read image files and load to RDD
        println("batchPath partition " + batchPath.getNumPartitions)
        println("batchPath count " + batchPath.count())
        if (!batchPath.isEmpty()) {
          println(batchPath.top(1).apply(0))
          // RDD[String] => RDD[ImageFeature]
          val dataset = ImageSet.rdd(batchPath.map(path => readFile(path)))
          // Resize image
          dataset -> ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)
          val output = model.predictImageSet(dataset)
          val visualizer = Visualizer(model.getConfig.labelMap, encoding = "jpg")
          val visualized = visualizer(output).toDistributed()
          val result = visualized.rdd.map(imageFeature =>
            (imageFeature.uri(), imageFeature[Array[Byte]](Visualizer.visualized))).collect()
          result.foreach(x => writeFile(params.outputFolder, x._1, x._2))
        }
      }
      ssc.start()
      ssc.awaitTermination()
      logger.info(s"labeled images are saved to ${params.outputFolder}")
    }
  }

  def readFile(path: String): ImageFeature = {
    println("Read image file " + path)
    val fspath = new Path(path)
    val fs = FileSystem.get(fspath.toUri, new Configuration())
    if (path.contains("hdfs")) {
      // Read HDFS image
      val instream = fs.open(fspath)
      val data = new Array[Byte](fs.getFileStatus(new Path(path))
        .getLen.toInt)
      instream.readFully(data)
      instream.close()
      ImageFeature.apply(data, null, path)
    } else {
      // Read local image
      ImageFeature(FileUtils.readFileToByteArray(new File(path)), uri = path)
    }
  }

  def writeFile(outPath: String, path: String, content: Array[Byte]): Unit = {
    val finalName = s"detection_${ path.substring(path.lastIndexOf("/") + 1,
      path.lastIndexOf(".")) }.jpg"
    val fspath = new Path(outPath, finalName)
    println("Writing image file " + fspath.toString)
    val fs = FileSystem.get(fspath.toUri, new Configuration())
    if (outPath.contains("hdfs")) {
      // Save to HDFS dir
      val outstream = fs.create(
        fspath,
        true)
      outstream.write(content)
      outstream.close()
    } else {
      // Save to local dir
      Utils.saveBytes(content,
        getOutPath(outPath, path, "jpg"), true)
    }
  }


  private def getOutPath(outPath: String, uri: String, encoding: String): String = {
    Paths.get(outPath,
      s"detection_${ uri.substring(uri.lastIndexOf("/") + 1,
        uri.lastIndexOf(".")) }.${encoding}").toString
  }
}

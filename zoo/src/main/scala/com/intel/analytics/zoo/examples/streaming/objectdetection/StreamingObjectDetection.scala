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

package com.intel.analytics.zoo.examples.streaming.objectdetection


import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.{ImageBytesToMat, ImageSet}
import com.intel.analytics.zoo.models.image.objectdetection.{ObjectDetector, Visualizer}
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

  case class PredictParam(streamingPath: String = "file:///tmp/zoo/streaming",
                          outputFolder: String = "file:///tmp/zoo/output",
                          model: String = "",
                          nPartition: Int = 1)

  val parser = new OptionParser[PredictParam]("Analytics Zoo Streaming Object Detection") {
    head("Analytics Zoo Streaming Object Detection")
    opt[String]('i', "streamingPath")
      .text("folder that used to store the streaming paths")
      .action((x, c) => c.copy(streamingPath = x))
      .required()
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
    opt[String]("model")
      .text("Analytics Zoo model path")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PredictParam()).foreach { params =>
      val sc = NNContext.initNNContext("Analytics Zoo Streaming Object Detection")
      val ssc = new StreamingContext(sc, Seconds(3))

      // Load pre-trained model
      val model = ObjectDetector.loadModel[Float](params.model)

      val lines = ssc.textFileStream(params.streamingPath)
      lines.foreachRDD { batchPath =>
        // Read image files and load to RDD
        logger.debug("batchPath partition " + batchPath.getNumPartitions)
        logger.debug("batchPath count " + batchPath.count())
        if (!batchPath.isEmpty()) {
          // RDD[String] => RDD[ImageFeature] => ImageSet
          val dataSet = ImageSet.rdd(batchPath.map(path => readFile(path)))
          // Resize image
          dataSet -> ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)
          val output = model.predictImageSet(dataSet)
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

  /**
   * Read files from local or remote dir
   * @param path file path
   * @return ImageFeature
   */
  def readFile(path: String): ImageFeature = {
    logger.info("Read image file " + path)
    val fspath = new Path(path)
    val fs = FileSystem.get(fspath.toUri, new Configuration())
    // Read local or remote image
    val inputStream = fs.open(fspath)
    val data = new Array[Byte](fs.getFileStatus(new Path(path))
      .getLen.toInt)
    inputStream.readFully(data)
    inputStream.close()
    ImageFeature.apply(data, null, path)
  }

  /**
   * Write files to local or remote dir
   * @param outPath output dir
   * @param path input file path
   * @param content file content
   */
  def writeFile(outPath: String, path: String, content: Array[Byte]): Unit = {
    val fspath = getOutPath(outPath, path, "jpg")
    logger.info("Writing image file " + fspath.toString)
    val fs = FileSystem.get(fspath.toUri, new Configuration())
    val outStream = fs.create(
      fspath,
      true)
    outStream.write(content)
    outStream.close()
  }

  /**
   * Generate final file path for predicted images
   * @param outPath
   * @param path
   * @param encoding
   * @return
   */
  private def getOutPath(outPath: String, path: String, encoding: String): Path = {
    val finalName = s"detection_${ path.substring(path.lastIndexOf("/") + 1,
      path.lastIndexOf(".")) }.${encoding}"
    new Path(outPath, finalName)
  }
}

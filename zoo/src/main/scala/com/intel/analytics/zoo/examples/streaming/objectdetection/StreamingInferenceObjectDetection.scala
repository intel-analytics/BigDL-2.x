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


import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.examples.streaming.objectdetection.StreamingObjectDetection.PredictParam
import com.intel.analytics.zoo.feature.image.{ImageBytesToMat, ImageChannelNormalize, ImageMatToTensor, ImageResize, ImageSet}
import com.intel.analytics.zoo.models.image.objectdetection.{LabelReader, ScaleDetection, Visualizer}
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser

object StreamingInferenceObjectDetection {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)

  val logger = Logger.getLogger(getClass)


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

      // Load pre-trained bigDL model
      val model = new InferenceModel(1)

      model.doLoadBigDL(params.model)

      val labelMap = LabelReader.apply("COCO")

      val lines = ssc.textFileStream(params.streamingPath)
      lines.foreachRDD { batchPath =>
        // Read image files and load to RDD
        logger.debug("batchPath partition " + batchPath.getNumPartitions)
        logger.debug("batchPath count " + batchPath.count())
        if (!batchPath.isEmpty()) {
          // RDD[String] => RDD[ImageFeature]
          val dataSet = ImageSet.rdd(batchPath.map(path => readFile(path)))
          // Pre-processing image
          val output = dataSet ->
            ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR) ->
            ImageResize(300, 300) ->
            ImageChannelNormalize(123f, 117f, 104f) ->
            ImageMatToTensor[Float]()
          val predict = output.toDistributed().rdd.map { img =>
            logger.debug("Begin Predict " + img.uri())
            // Add one more dim because of batch requirement of model
            val predict = model
              .doPredict(img.apply[Tensor[Float]](ImageFeature.imageTensor)
              .addSingletonDimension())
            logger.debug("Finish Predict " + img.uri())
            img(ImageFeature.predict) = predict.toTensor[Float].apply(1)
            img
          }
          // ROI to box
          val boxedPredict = ImageSet.rdd(predict) -> ScaleDetection()
          logger.info("Begin visualizing box for image")
          val visualizer = Visualizer(labelMap, encoding = "jpg")
          val visualized = visualizer(boxedPredict).toDistributed()
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
   * Read image files from local or remote file system
   * @param path file path
   * @return ImageFeature
   */
  def readFile(path: String): ImageFeature = {
    val fspath = new Path(path)
    logger.info("Read image file " + path)
    val fs = FileSystem.newInstance(fspath.toUri, new Configuration())
    // Read local or remote image
    val inputStream = fs.open(fspath)
    val data = new Array[Byte](fs.getFileStatus(new Path(path))
      .getLen.toInt)
    inputStream.readFully(data)
    inputStream.close()
    ImageFeature.apply(data, null, path)
  }

  /**
   * Write image file to local or remote file system
   * @param outPath accessible output dir. Output file will be
   *                {outPath}/detection_{path}
   * @param path    original image path
   * @param content image Array[byte]
   */
  def writeFile(outPath: String, path: String, content: Array[Byte]): Unit = {
    val fspath = getOutPath(outPath, path, "jpg")
    logger.info("Writing image file " + fspath.toString)
    val fs = FileSystem.newInstance(fspath.toUri, new Configuration())
    val outStream = fs.create(
      fspath,
      true)
    outStream.write(content)
    outStream.close()
  }

  /**
   * Generate final file path for predicted images
   * @param outPath  accessible output dir
   * @param path     original image path
   * @param encoding image encoding, e.g., JPG
   * @return Final file path, e.g., {outPath}/detection_{path}
   */
  private def getOutPath(outPath: String, path: String, encoding: String): Path = {
    val finalName = s"detection_${ path.substring(path.lastIndexOf("/") + 1,
      path.lastIndexOf(".")) }.${encoding}"
    new Path(outPath, finalName)
  }
}

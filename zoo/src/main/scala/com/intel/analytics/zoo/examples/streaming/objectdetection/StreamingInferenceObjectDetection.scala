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


<<<<<<< Updated upstream
import com.intel.analytics.bigdl.dataset.MiniBatch
=======
import com.intel.analytics.bigdl.dataset.SampleToMiniBatch
>>>>>>> Stashed changes
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.examples.streaming.objectdetection.StreamingObjectDetection.PredictParam
import com.intel.analytics.zoo.feature.FeatureSet
<<<<<<< Updated upstream
import com.intel.analytics.zoo.feature.image.roi.RoiRecordToFeature
=======
>>>>>>> Stashed changes
import com.intel.analytics.zoo.feature.image.{ImageBytesToMat, ImageMatToFloats, ImageResize, ImageSet}
import com.intel.analytics.zoo.models.image.objectdetection.LabelReader
import com.intel.analytics.zoo.models.image.objectdetection.Visualizer
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.ByteRecord
import com.intel.analytics.zoo.models.image.objectdetection.ssd.{RoiImageToSSDBatch, SSDMiniBatch}
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser

object StreamingInferenceObjectDetection {
//  Logger.getLogger("org").setLevel(Level.ERROR)
//  Logger.getLogger("akka").setLevel(Level.ERROR)
//  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

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
      val model = new InferenceModel(4)

      // TODO Load labelMap, need API help
      val labelMap = LabelReader.apply("COCO")

      val lines = ssc.textFileStream(params.streamingPath)
      val visualizer = Visualizer(labelMap, encoding = "jpg")
      lines.foreachRDD { batchPath =>
        // Read image files and load to RDD
        logger.debug("batchPath partition " + batchPath.getNumPartitions)
        logger.debug("batchPath count " + batchPath.count())
        if (!batchPath.isEmpty()) {
          // RDD[String] => RDD[ImageFeature]
          val dataSet = ImageSet.rdd(batchPath.map(path => readFile(path)))
          // Resize image
          val output = dataSet -> ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR) ->
            ImageResize(300, 300) ->
            ImageMatToFloats()
          val batched = output.toDataSet() -> SampleToMiniBatch[Float](4)
          val predicts = batched.toDistributed().data(false).map { minibatch =>
            logger.info("Begin Predict ")
            // Add one more dim because of batch requirement of model
            model.doPredict(minibatch.getInput())
          }
//            // Add one more dim because of batch requirement of model
//
//            logger.info("Begin visualizing box for image " + img.uri())
//            // TODO Visualizer also need some changes
//            val result = visualizer.visualize(OpenCVMat.fromImageBytes(img.bytes()),
//              output.toTensor)
//            val resultImg = OpenCVMat.imencode(result, "jpg")
//            writeFile(params.outputFolder, img.uri(), resultImg)
        }
      }
      ssc.start()
      ssc.awaitTermination()
      logger.info(s"labeled images are saved to ${params.outputFolder}")
    }
  }

  /**
    * Read image files from local or remote filgde system
    * @param path file path
    * @return ByteRecord
    */
  def readFile(path: String): ByteRecord = {
    logger.info("Read image file " + path)
    val data = Utils.readBytes(path)
    ByteRecord.apply(data, path)
  }

  /**
    * Write image file to local or remote file system
    * @param outPath output dir
    * @param path input file path
    * @param content file content
    */
  def writeFile(outPath: String, path: String, content: Array[Byte]): Unit = {
    val fspath = getOutPath(outPath, path, "jpg")
    logger.info("Writing image file " + fspath.toString)
    Utils.saveBytes(content, fspath.toString, true)
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

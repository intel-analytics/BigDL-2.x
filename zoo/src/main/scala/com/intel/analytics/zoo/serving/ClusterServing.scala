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

package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.dataset.SampleToMiniBatch
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.models.image.imageclassification.{LabelOutput, LabelReader}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.{EngineRef, KerasUtils}
import com.intel.analytics.zoo.serving.utils.Result
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, ImageClassification, ObjectDetection}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.streaming.DataStreamWriter
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SaveMode}
import org.opencv.imgcodecs.Imgcodecs


object ClusterServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {

    val helper = new ClusterServingHelper()
    helper.init(args)
    val coreNumber = EngineRef.getCoreNumber()
    val eType = EngineRef.getEngineType()
    logger.info("Engine Type is " + eType)
    logger.info("Core number is running at " + coreNumber.toString)


    val model = helper.loadInferenceModel(coreNumber)

    val spark = helper.getSparkSession()

    logger.info(s"connected to redis ${spark.conf.get("spark.redis.host")}:${spark.conf.get("spark.redis.port")}")
    val batchSize = helper.batchSize
    val topN = helper.topN

    val images = spark
      .readStream
      .format("redis")
      .option("stream.keys", "image_stream")
      .option("stream.read.batch.size", batchSize.toString)
      .option("stream.parallelism", EngineRef.getNodeNumber())
      .schema(StructType(Array(
        StructField("id", StringType),
        StructField("path", StringType),
        StructField("image", StringType)
      )))
      .load()
    val foreachBatchMethod = images.writeStream.getClass()
      .getMethods().filter(_.getName() == "foreachBatch")
        .filter(_.getParameterTypes()(0).getName() == "scala.Function2")(0)
    val query = foreachBatchMethod.invoke(images.writeStream,
      (batchDF: DataFrame, batchId: Long) => {
        logger.info("getting batch")
        val batchImage = batchDF.rdd.map { image =>
          val bytes = java.util
            .Base64.getDecoder.decode(image.getAs[String]("image"))
          val path = image.getAs[String]("path")

          ImageFeature.apply(bytes, null, path)
        }
        val imageSet = ImageSet.rdd(batchImage)
        imageSet.rdd.persist()
        logger.info("Micro batch size " + imageSet.rdd.count().toString)

        val inputs = imageSet ->
          ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR) ->
//          ImageResize(256, 256) ->
//          ImageCenterCrop(224, 224) ->
          ImageMatToTensor(shareBuffer = false) ->
          ImageSetToSample()

        val start = System.nanoTime()

        val bcModel = model.value
        logger.info("start predict")
        // switch mission here, e.g. image classification, object detection
        var result: RDD[Result] = null
        if (helper.params.task == "image-classification") {
          result = ImageClassification.getResult(inputs, bcModel, helper)
        } else {
          throw new Error("Your task specified is " + helper.params.task +
          "Currently Cluster Serving only support image-classification")
        }

        // Output results
        val resDf = spark.createDataFrame(result)
        resDf.write
          .format("org.apache.spark.sql.redis")
          .option("table", "result")

          .mode(SaveMode.Append).save()

        val latency = System.nanoTime() - start
        logger.info(s"Predict latency is ${latency / 1e6} ms")
        imageSet.rdd.unpersist()
      }
    ).asInstanceOf[DataStreamWriter[Row]].start()
    query.awaitTermination()
  }
}


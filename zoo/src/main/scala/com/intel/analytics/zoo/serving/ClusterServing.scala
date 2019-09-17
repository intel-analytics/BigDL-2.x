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
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, ImageClassification, ObjectDetection}
import com.intel.analytics.zoo.utils._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.ZippedPartitionsWithLocalityRDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SaveMode}
import org.opencv.imgcodecs.Imgcodecs


object ClusterServing {


  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {

//    val param = parser.parse(args, RedisParams()).get

    val helper = new ClusterServingHelper()
    helper.init(args)
    val model = helper.loadInferenceModel()

    val coreNumber = EngineRef.getCoreNumber()

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
    val query = images
      .writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) => {
        logger.info("getting batch")
        val batchImage = batchDF.rdd.map { image =>
          val bytes = java.util
            .Base64.getDecoder.decode(image.getAs[String]("image"))
          val path = image.getAs[String]("path")

          ImageFeature.apply(bytes, null, path)
        }
        val imageSet = ImageSet.rdd(batchImage)
        imageSet.rdd.persist()
        val st = imageSet.rdd.collect()

        val inputs = imageSet ->
          ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR) ->
//          ImageResize(256, 256) ->
//          ImageCenterCrop(224, 224) ->
          ImageMatToTensor(shareBuffer = false) ->
          ImageSetToSample()

        val start = System.nanoTime()

        val bcModel = model.value

        // switch mission here, e.g. image classification, object detection
        val result = ImageClassification.getResult(inputs, bcModel, helper)


        // Output results
        val resDf = spark.createDataFrame(result)
        resDf.write
          .format("org.apache.spark.sql.redis")
          .option("table", "result")
          .mode(SaveMode.Append).save()

        val latency = System.nanoTime() - start
        logger.info(s"Predict latency is ${latency / 1e6} ms")
      }
    }.start()
    query.awaitTermination()
  }
}


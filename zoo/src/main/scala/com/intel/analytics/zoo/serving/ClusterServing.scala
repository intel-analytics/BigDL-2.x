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


import java.io.FileWriter
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, ImageClassification}
import com.intel.analytics.zoo.utils.ImageProcessing
import org.apache.log4j.{Level, Logger}

import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SaveMode}



object ClusterServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {

//    System.setProperty("bigdl.engineType", "mkldnn")
    val helper = new ClusterServingHelper()
    helper.initArgs(args)
    helper.initContext()

    val model = helper.loadInferenceModel()


    val spark = helper.getSparkSession()

    logger.info(s"connected to redis " +
      s"${spark.conf.get("spark.redis.host")}:${spark.conf.get("spark.redis.port")}")
    val batchSize = helper.batchSize
    val topN = helper.topN

    val images = spark
      .readStream
      .format("redis")
      .option("stream.keys", "image_stream")
      .option("stream.read.batch.size", 512)
      .option("stream.parallelism", EngineRef.getNodeNumber())
      .schema(StructType(Array(
        StructField("id", StringType),
        StructField("path", StringType),
        StructField("image", StringType)
      )))
      .load()

    val query = images.writeStream.foreachBatch{ (batchDF: DataFrame, batchId: Long) =>
      batchDF.persist()
      val microBatchSize = batchDF.count()
      logger.info("Micro batch size " + microBatchSize.toString)
      val pathBytesRDDChunk = batchDF.rdd.map { image =>
        // single thread preprocessing here
        val bytes = ImageProcessing.bytesToBGRTensor(java.util
          .Base64.getDecoder.decode(image.getAs[String]("image")))
        val path = image.getAs[String]("path")

        (path, bytes)
      }
      val pathBytesRDD = if (helper.blasFlag == true) {
        pathBytesRDDChunk.repartition(helper.nodeNum * helper.coreNum)
      } else {
        pathBytesRDDChunk
      }

      // variables needed to be serialized listed below
      val modelType = helper.modelType

      val res = pathBytesRDD.mapPartitions(pathBytes => {
        val localModel = model.value
        pathBytes.grouped(batchSize).map(pathByteBatch => {
          val thisBatchSize = pathByteBatch.size
          val t = Tensor[Float](batchSize, 3, 224, 224)

          (0 until thisBatchSize).foreach(i => t.select(1, i + 1).copy(pathByteBatch(i)._2))

          val x = if (modelType == "tensorflow") {
            t.transpose(2, 3)
              .transpose(3, 4).contiguous()
          } else {
            t
          }

          val start = System.nanoTime()
          val result = localModel.doPredict(x)
          val end = System.nanoTime()
          println(s"elapsed ${(end - start) / 1e9} s")
          result
        })

      }).count()
      logger.info("Micro batch predict ended")
    }.start()
//    val foreachBatchMethod = images.writeStream.getClass()
//      .getMethods().filter(_.getName() == "foreachBatch")
//        .filter(_.getParameterTypes()(0).getName() == "scala.Function2")(0)
//    val query = foreachBatchMethod.invoke(images.writeStream,
//      (batchDF: DataFrame, batchId: Long) => {
//        batchDF.persist()
//        logger.info("getting batch" + batchId)
//        logger.info(s"num of partition: ${batchDF.rdd.partitions.size}")
//
//        val partitionedBatchDF = batchDF.repartition(EngineRef.getNodeNumber())
////        partitionedBatchDF.foreachPartition(_ => print(""))
//
//        val batchImage = partitionedBatchDF.rdd.map { image =>
//          val bytes = java.util
//            .Base64.getDecoder.decode(image.getAs[String]("image"))
//          val path = image.getAs[String]("path")
//
//          ImageFeature.apply(bytes, null, path)
//        }
//        val imageSet = ImageSet.rdd(batchImage)
//        imageSet.rdd.persist()
//        val microBatchSize = imageSet.rdd.count()
//        logger.info("Micro batch size " + microBatchSize.toString)
//        if (!imageSet.rdd.isEmpty()) {
//          val inputs = imageSet ->
//            ImageBytesToMat(imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR) ->
//            //          ImageResize(256, 256) ->
//            //          ImageCenterCrop(224, 224) ->
//            ImageMatToTensor(shareBuffer = false) ->
//            ImageSetToSample()
//
//          val start = System.nanoTime()
//
//          val bcModel = model.value
//
//          logger.info("start predict")
//          // switch mission here, e.g. image classification, object detection
//          var result: RDD[Result] = null
//          if (helper.params.task == "image-classification") {
//            result = ImageClassification.getResult(inputs, bcModel, helper)
//          } else {
//            throw new Error("Your task specified is " + helper.params.task +
//              "Currently Cluster Serving only support image-classification")
//          }
//          val latency = System.nanoTime() - start
//          logger.info(s"Predict latency is ${latency / 1e6} ms")
//
//
//          val fw = new FileWriter("/tmp/tp.txt", true)
//          val throughput = microBatchSize.toFloat / (latency / 1e9)
//          fw.write(throughput.toString + "\n")
//          fw.close()
//
//
//          // Output results
//          val resDf = spark.createDataFrame(result)
//          resDf.write
//            .format("org.apache.spark.sql.redis")
//            .option("table", "result")
//
//            .mode(SaveMode.Append).save()
//        }
//        imageSet.rdd.unpersist()
//      }
//    ).asInstanceOf[DataStreamWriter[Row]].start()

    query.awaitTermination()
  }
}


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


import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.inference.InferenceSummary
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, PostProcessing, TensorUtils}
import com.intel.analytics.zoo.utils.ImageProcessing
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SaveMode}
import redis.clients.jedis.Jedis



object ClusterServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)



  def main(args: Array[String]): Unit = {

    val helper = new ClusterServingHelper()
    helper.initArgs()
    helper.initContext()

    val logger = helper.logger
    logger.info("Engine running at " + EngineRef.getEngineType())

    val model = helper.loadInferenceModel()
    val bcModel = helper.sc.broadcast(model)
    model.setInferenceSummary(null)


    val spark = helper.getSparkSession()

    logger.info(s"connected to redis " +
      s"${spark.conf.get("spark.redis.host")}:${spark.conf.get("spark.redis.port")}")

    // variables needed to be serialized listed below
    val batchSize = helper.batchSize
    val topN = helper.topN
    val coreNum = helper.coreNum
    val nodeNum = helper.nodeNum
    val modelType = helper.modelType

    val C = helper.dataShape(0)
    val W = helper.dataShape(1)
    val H = helper.dataShape(2)

    val images = spark
      .readStream
      .format("redis")
      .option("stream.keys", "image_stream")
      .option("stream.read.batch.size", batchSize)
      .option("stream.parallelism", nodeNum)
      .schema(StructType(Array(
        StructField("uri", StringType),
        StructField("image", StringType)
      )))
      .load()

    var totalCnt: Int = 0

    // redis stream control
    val redisDB = new Jedis(helper.redisHost, helper.redisPort.toInt)

    val query = images.writeStream.foreachBatch{ (batchDF: DataFrame, batchId: Long) =>
      batchDF.persist()
      val microBatchSize = batchDF.count()
      logger.info("Micro batch size " + microBatchSize.toString)

      if (microBatchSize != 0) {

        val microBatchStart = System.nanoTime()
        val resultPartitions = if (helper.blasFlag) {
          // if backend is raw BLAS, no multi-thread could used for forward
          // BLAS is only valid in BigDL backend
          // Thus no shape specific change is needed
          batchDF.rdd.mapPartitions(pathBytes => {
            pathBytes.grouped(coreNum).flatMap(pathBytesBatch => {
              pathBytesBatch.indices.toParArray.map(i => {
                val tensors = ImageProcessing.bytesToBGRTensor(java.util
                  .Base64.getDecoder.decode(pathBytesBatch(i).getAs[String]("image")))
                val path = pathBytesBatch(i).getAs[String]("uri")

                val localPartitionModel = bcModel.value
                val result = localPartitionModel.doPredict(tensors.addSingletonDimension()).toTensor

                val value = PostProcessing.getInfofromTensor(topN, result)

                (path, value)
              })
            })
          })
        } else {
          val pathBytesChunk = batchDF.rdd.mapPartitions(pathBytes => {
            pathBytes.grouped(coreNum).flatMap(pathBytesBatch => {
              pathBytesBatch.indices.toParArray.map(i => {
                val tensors = ImageProcessing.bytesToBGRTensor(java.util
                  .Base64.getDecoder.decode(pathBytesBatch(i).getAs[String]("image")))
                val path = pathBytesBatch(i).getAs[String]("uri")

                (path, tensors)
              })
            })
          })
          pathBytesChunk.mapPartitions(pathBytes => {
            val localModel = bcModel.value
            pathBytes.grouped(batchSize).flatMap(pathByteBatch => {
              val thisBatchSize = pathByteBatch.size
              val t = Tensor[Float](batchSize, C, W, H)

              (0 until thisBatchSize).foreach(i => t.select(1, i + 1).copy(pathByteBatch(i)._2))

              val x = if (modelType == "tensorflow") {
                t.transpose(2, 3)
                  .transpose(3, 4).contiguous()
              } else if (modelType == "openvino") {
                t.addSingletonDimension()
              } else {
                t
              }

              val result = if (modelType == "openvino") {
                localModel.doPredict(x).toTensor.squeeze()
              } else {
                localModel.doPredict(x).toTensor
              }

              // result post processing starts here
              // move below code into wrapper if new functions
              // e.g. object detection is added

              (0 until thisBatchSize).toParArray.map(i => {
                val value = PostProcessing.getInfofromTensor(topN,
                  result.select(1, i + 1).squeeze())
                (pathByteBatch(i)._1, value)
              })

            })
          })
        }


        val resDf = spark.createDataFrame(resultPartitions)
        resDf.write
          .format("org.apache.spark.sql.redis")
          .option("table", "result")
          .mode(SaveMode.Append).save()


        totalCnt += microBatchSize.toInt
        val microBatchEnd = System.nanoTime()
        val microBatchLatency = (microBatchEnd - microBatchStart) / 1e9
        val microBatchThroughPut = (microBatchLatency / microBatchSize).toFloat
        if (model.inferenceSummary != null) {
          model.inferenceSummary.addScalar(
            "Micro Batch Throughput", microBatchThroughPut, batchId)
          model.inferenceSummary.addScalar(
            "Partition Number", batchDF.rdd.partitions.size, batchId)
          model.inferenceSummary.addScalar(
            "Total Records Number", totalCnt, batchId)
        }

        logger.info("Micro batch predict ended, time elapsed " + microBatchLatency.toString)

        redisDB.xtrim("image_stream", redisDB.xlen("image_stream"), true)

      }

    }.start()

    query.awaitTermination()
  }
}

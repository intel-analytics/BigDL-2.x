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
import com.intel.analytics.zoo.pipeline.inference.{InferenceModel, InferenceSummary}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, ClusterServingListener, PostProcessing, TensorUtils}
import com.intel.analytics.zoo.utils.ImageProcessing
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, types}
import redis.clients.jedis.Jedis


object ClusterServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)


  case class Record(uri: String, value: String)
  var batchSize: Int = 4
  var topN: Int = 1
  var coreNum: Int = 1
  var nodeNum: Int = 1
  var modelType: String = null
  var blasFlag: Boolean = false

  var C: Int = 3
  var W: Int = 224
  var H: Int = 224

  def loadSerialParams(helper: ClusterServingHelper): Unit = {
    batchSize = helper.batchSize
    topN = helper.topN
    coreNum = helper.coreNum
    nodeNum = helper.nodeNum
    modelType = helper.modelType
    blasFlag = helper.blasFlag

    C = helper.dataShape(0)
    W = helper.dataShape(1)
    H = helper.dataShape(2)
  }


  def main(args: Array[String]): Unit = {

    val helper = new ClusterServingHelper()
    helper.initArgs()
    helper.initContext()

    val logger = helper.logger
    logger.info("Engine running at " + EngineRef.getEngineType())

    var model: InferenceModel = null
    var bcModel: Broadcast[InferenceModel] = null

    loadSerialParams(helper)
    model = helper.loadInferenceModel()
    bcModel = helper.sc.broadcast(model)
//    if (helper.logSummaryFlag) model.setInferenceSummary(
//      InferenceSummary(".", helper.dateTime + "-ClusterServing"))

    model.setInferenceSummary(
      InferenceSummary(".", helper.dateTime + "-ClusterServing"))

    val spark = helper.getSparkSession()

    logger.info(s"connected to redis " +
      s"${spark.conf.get("spark.redis.host")}:${spark.conf.get("spark.redis.port")}")

    // variables needed to be serialized listed below

    loadSerialParams(helper)

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
    var timeStamp: Int = 0

    // redis stream control
    val redisDB = new Jedis(helper.redisHost, helper.redisPort.toInt)

    val query = images.writeStream.foreachBatch{ (batchDF: DataFrame, batchId: Long) =>

//      if (helper.updateConfig()) {
//        loadSerialParams(helper)
//        if (bcModel != null) bcModel.destroy()
//        if (model != null) model.doRelease()
//        model = helper.loadInferenceModel()
//        bcModel = helper.sc.broadcast(model)
//        if (helper.logSummaryFlag) model.setInferenceSummary(
//          InferenceSummary(".", helper.dateTime + "-ClusterServing"))
//      }

      batchDF.persist()

      val microBatchSize = batchDF.count()
      logger.info("Micro batch size " + microBatchSize.toString)

      if (microBatchSize != 0) {
        val previousLen = redisDB.xlen("image_stream")

        val microBatchStart = System.nanoTime()

        val resultPartitions = if (blasFlag) {
          // if backend is raw BLAS, no multi-thread could used for forward
          // BLAS is only valid in BigDL backend
          // Thus no shape specific change is needed
          batchDF.rdd.mapPartitions(pathBytes => {
            pathBytes.grouped(coreNum).flatMap(pathBytesBatch => {
              pathBytesBatch.indices.toParArray.map(i => {
//                val raw = pathBytesBatch(i).getString(0)
//                if(raw != )
                val tensors = ImageProcessing.bytesToBGRTensor(java.util
                  .Base64.getDecoder.decode(pathBytesBatch(i).getAs[String]("image")))
                val path = pathBytesBatch(i).getAs[String]("uri")

                val localPartitionModel = bcModel.value
                val result = localPartitionModel.doPredict(tensors.addSingletonDimension()).toTensor

                val value = PostProcessing.getInfofromTensor(topN, result)

                Record(path, value)
              })
            })
          })
        } else {
          val pathBytesChunk = batchDF.rdd.filter(_.getAs[String]("uri") != "STOP").mapPartitions(pathBytes => {
            pathBytes.grouped(coreNum).flatMap(pathBytesBatch => {
              pathBytesBatch.indices.toParArray.map(i => {
                val row = pathBytesBatch(i)
                val path = row.getAs[String]("uri")
                val tensors = ImageProcessing.bytesToBGRTensor(java.util
                  .Base64.getDecoder.decode(row.getAs[String]("image")))
                (path, tensors)

//                val tensors = ImageProcessing.bytesToBGRTensor(java.util
//                  .Base64.getDecoder.decode(pathBytesBatch(i).getAs[String]("image")))
//                val path = pathBytesBatch(i).getAs[String]("uri")


              })
            })
          })
          pathBytesChunk.mapPartitions(pathBytes => {
            val localModel = bcModel.value
            pathBytes.grouped(batchSize).flatMap(pathByteBatch => {
              val thisBatchSize = pathByteBatch.size
              val t = Tensor[Float](batchSize, C, W, H)

              (0 until thisBatchSize).toParArray
                .foreach(i => t.select(1, i + 1).copy(pathByteBatch(i)._2))

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
                Record(pathByteBatch(i)._1, value)
              })

            })
          })
        }


        val resDf = spark.createDataFrame(resultPartitions)

        var errFlag: Boolean = true
        while (errFlag) {
          try {
            resDf.write
              .format("org.apache.spark.sql.redis")
              .option("table", "result")
              .option("key.column", "uri")
              .mode(SaveMode.Append).save()
            println("result saved at " + resDf.count())
            redisDB.xtrim("image_stream",
              redisDB.xlen("image_stream") - previousLen, true)
            println("Xtrim completed")
            errFlag = false
          }

          catch {
            case e: redis.clients.jedis.exceptions.JedisDataException =>
              errFlag = true
              val errMsg = "not enough space in redis to write: " + e.toString
              println(errMsg)
              helper.logFile.write(errMsg)
              Thread.sleep(3000)
            case e: Exception => {
              errFlag = true
              val errMsg = "unable to handle exception: " + e.toString
              println(errMsg)
              Thread.sleep(3000)
            }

          }
        }

        val microBatchEnd = System.nanoTime()
        val microBatchLatency = (microBatchEnd - microBatchStart) / 1e9
        val microBatchThroughPut = (microBatchSize / microBatchLatency).toFloat

        totalCnt += microBatchSize.toInt

        if (model.inferenceSummary != null) {
          (timeStamp until timeStamp + microBatchLatency.toInt).foreach( time => {
            model.inferenceSummary.addScalar(
              "Serving Throughput", microBatchThroughPut, time)
          })
//          model.inferenceSummary.addScalar(
//            "Micro Batch Throughput", microBatchThroughPut, batchId)

          model.inferenceSummary.addScalar(
            "Total Records Number", totalCnt, batchId)
        }

        timeStamp += microBatchLatency.toInt

        logger.info(microBatchSize +
          " inputs predict ended, time elapsed " + microBatchLatency.toString)

        if (helper.checkStop()) {
          println("stop signal detected")
          System.exit(0)
        }
      }
    }

    val servingQuery = query.start()

    ClusterServingListener.listenTermination(helper, servingQuery)
    servingQuery.awaitTermination()

    assert(spark.streams.active.isEmpty)

//    while (true) {
//      Thread.sleep(1000)
//      if (helper.checkStop()) query.stop()
//    }
  }
}

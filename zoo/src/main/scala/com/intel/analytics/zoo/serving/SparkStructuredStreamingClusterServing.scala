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
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.pipeline.inference.{InferenceModel, InferenceSummary}
import com.intel.analytics.zoo.serving.engine.InferenceSupportive
import com.intel.analytics.zoo.serving.utils._
import com.intel.analytics.zoo.serving.pipeline._
import com.intel.analytics.zoo.serving.preprocessing.DataType
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SaveMode}
import redis.clients.jedis.Jedis

import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Failure, Success}
import org.apache.spark.util.LongAccumulator

import scala.collection.mutable.ArrayBuffer

object SparkStructuredStreamingClusterServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo.feature.image").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)


  case class Record(uri: String, value: String)


  def main(args: Array[String]): Unit = {

    val helper = new ClusterServingHelper()
    helper.initArgs()
    helper.initContext()

    /**
     * Variables need to be serialized are listed below
     * Take them from helper in advance for later execution
     */
    val coreNum = helper.coreNum
    val nodeNum = helper.nodeNum
    val modelType = helper.modelType
    val blasFlag = helper.blasFlag
//    val dataType = helper.dataType
    val dataShape = helper.dataShape

    val (streamKey, dataField) = ("serving_stream", "data")

    val filter = helper.filter

    /**
     * chwFlag is to set image input of CHW or HWC
     * if true, the format is CHW
     * else, the format is HWC
     *
     * Note that currently CHW is commonly used
     * and HWC is often used in Tensorflow models
     */
    val chwFlag = if (modelType.startsWith("tensorflow")) {
      false
    } else {
      true
    }

    val logger = helper.logger

    /**
     * For cut input stream, this variable is to avoid
     * unsafe cut of input stream, if cut current batch
     * there is an interval between get and cut, this would
     * affect the result of correctness, some new data might be cut
     */

    var model: InferenceModel = null
    var bcModel: Broadcast[InferenceModel] = null

    model = helper.loadInferenceModel()
    bcModel = helper.sc.broadcast(model)

    model.setInferenceSummary(
      InferenceSummary("./TensorboardEventLogs", helper.dateTime + "-ClusterServing"))

    val spark = helper.getSparkSession()

    logger.info(s"connected to redis " +
      s"${spark.conf.get("spark.redis.host")}:${spark.conf.get("spark.redis.port")}")


    val inputData = spark
      .readStream
      .format("redis")
      .option("stream.keys", streamKey)
      .option("stream.read.batch.size", coreNum)
      .option("stream.parallelism", nodeNum)
      .schema(StructType(Array(
        StructField("uri", StringType),
        StructField(dataField, StringType)
      )))
      .load()

    var totalCnt: Int = 0
    var timeStamp: Int = 0

    // redis stream control
    val jedis = new Jedis(helper.redisHost, helper.redisPort.toInt)
    val inputThreshold = 0.6 * 0.8
    val cutRatio = 0.5

    val acc = new LongAccumulator()
    helper.sc.register(acc)
    val serParams = new SerParams(helper, false)

    val query = inputData.writeStream.foreachBatch{ (batchDF: DataFrame, batchId: Long) =>

      /**
       * This is reserved for future dynamic loading model
       */
      val redisInfo = RedisUtils.getMapFromInfo(jedis.info())

      if (redisInfo("used_memory").toLong >=
        redisInfo("maxmemory").toLong * inputThreshold) {
        jedis.xtrim(streamKey,
          (jedis.xlen(streamKey) * cutRatio).toLong, true)
      }

      batchDF.persist()


      if (!batchDF.isEmpty) {
        /**
         * The streaming may be triggered somehow and it is possible
         * to get an empty batch
         *
         * If the batch is not empty, start preprocessing and predict here
         */
        val microBatchStart = System.nanoTime()
        acc.reset()
        RedisUtils.checkMemory(jedis, inputThreshold, cutRatio)

        /**
         * The streaming may be triggered somehow and it is possible
         * to get an empty batch
         *
         * If the batch is not empty, start preprocessing and predict here
         */
        val x = batchDF.rdd
        val preProcessed = x.mapPartitions(it => {
          val pre = new PreProcessing(serParams)
          it.grouped(serParams.coreNum).flatMap(itemBatch => {
            acc.add(itemBatch.size)
            itemBatch.indices.toParArray.map(i => {
              val uri = itemBatch(i).getAs[String]("uri")
              val tensor = pre.decodeArrowBase64(itemBatch(i).getAs[String]("data"))
              (uri, tensor)
            })
          })
        })

        /**
         * Engine type controlling, for different engine type,
         * different partitioning and batching scheduling is used
         */
        val postProcessed = if (blasFlag) {
          /**
           * In BLAS mode, every model could predict only using
           * a single thread, besides, batch size usually is not
           * over 64 in serving to achieve good latency. Thus, no
           * batching is required if the machine has over about 30 cores.           *
           */
          preProcessed.mapPartitions(it => {
            serParams.model = bcModel.value
            it.grouped(serParams.coreNum).flatMap(itemBatch => {
              InferenceSupportive.multiThreadInference(itemBatch.toIterator, serParams)
            })
          })


        } else {
          /**
           * In Normal mode, every model will use multiple thread to
           * achieve best latency. Thus, we only use a single model to
           * do sequential predict, maximizing the latency performance
           * and minimizing the memory usage.
           */
          preProcessed.mapPartitions(it => {
            serParams.model = bcModel.value
            it.grouped(serParams.coreNum).flatMap(itemBatch => {
              InferenceSupportive.multiThreadInference(itemBatch.toIterator, serParams)
            })
          })
        }
        val resRdd = postProcessed.map(x => Record(x._1, x._2))
        /**
         * Predict ends, start writing data to output queue
         */
        val resDf = spark.createDataFrame(resRdd)

        var errFlag: Boolean = true

        /**
         * Block the inference if there is no space to write
         * Will continuously try write the result, if not, keep blocking
         * The input stream may accumulate to very large because no records
         * will be consumed, however the stream size would be controlled
         * on Cluster Serving API side.
         */
        while (errFlag) {
          try {
            resDf.write
              .format("org.apache.spark.sql.redis")
              .option("table", "result")
              .option("key.column", "uri")
              .option("iterator.grouping.size", coreNum)
              .mode(SaveMode.Append).save()

            errFlag = false
          }

          catch {
            case e: redis.clients.jedis.exceptions.JedisDataException =>
              errFlag = true
              val errMsg = "not enough space in redis to write: " + e.toString
              logger.info(errMsg)
              Thread.sleep(3000)
            case e: java.lang.InterruptedException =>
              /**
               * If been interrupted by stop signal, do nothing
               * End the streaming until this micro batch process ends
               */
              logger.info("Stop signal received, will exit soon.")
            case e: Exception =>
              errFlag = true
              val errMsg = "unable to handle exception: " + e.toString
              logger.info(errMsg)
              Thread.sleep(3000)

          }
        }

        /**
         * Count the statistical data and write to summary
         */
        val microBatchEnd = System.nanoTime()

        val microBatchLatency = (microBatchEnd - microBatchStart) / 1e9
        val microBatchThroughPut = (acc.value / microBatchLatency).toFloat
        logger.info(s"Inferece end. Input size ${acc.value}. " +
          s"Latency $microBatchLatency, Throughput $microBatchThroughPut")

        totalCnt += acc.value.toInt

        val lastTimeStamp = timeStamp
        timeStamp += microBatchLatency.toInt + 1

        AsyncUtils.writeServingSummay(model,
          lastTimeStamp, timeStamp, totalCnt, microBatchThroughPut)
          .onComplete{
            case Success(_) => None

            case Failure(exception) => logger.info(s"$exception, " +
              s"write summary fails, please check.")
          }



      }
    }

    /**
     * Start the streaming, and listen to stop signal
     * The stop signal will be listened by another thread
     * with interval of 1 second.
     */
    val servingQuery = query.start()

//    ClusterServingManager.listenTermination(helper, servingQuery)

    servingQuery.awaitTermination()

    assert(spark.streams.active.isEmpty)
    System.exit(0)
  }
}

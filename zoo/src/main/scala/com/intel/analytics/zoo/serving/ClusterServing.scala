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
import com.intel.analytics.zoo.pipeline.inference.{InferenceModel, InferenceSummary}
import com.intel.analytics.zoo.serving.utils._
import com.intel.analytics.zoo.serving.InferenceStrategy
import com.intel.analytics.zoo.serving.spark.ServingReceiver
import com.redislabs.provider.redis.streaming.ConsumerConfig
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SaveMode}
import org.apache.spark.streaming.{Duration, StreamingContext}
import redis.clients.jedis.Jedis
import com.redislabs.provider.redis.streaming._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.{DoubleAccumulator, LongAccumulator}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Failure, Success}


object ClusterServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.ERROR)


  case class Record(uri: String, value: String)
  class Params(_coreNum: Int,
               _filter: String,
               _chwFlag: Boolean,
               _C: Int,
               _H: Int,
               _W: Int,
               _modelType: String) extends Serializable {
    val coreNum = _coreNum
    val filter = _filter
    val chwFlag = _chwFlag
    val C = _C
    val H = _H
    val W = _W
    val modelType = _modelType
  }


  def main(args: Array[String]): Unit = {

    val helper = new ClusterServingHelper()
    helper.initArgs()
    helper.initContext()

    /**
     * Variables need to be serialized are listed below
     * Take them from helper in advance for later execution
     */
    val batchSize = helper.batchSize
    val topN = helper.topN
    val coreNum = helper.coreNum
    val nodeNum = helper.nodeNum
    val modelType = helper.modelType
    val blasFlag = helper.blasFlag

    val C = helper.dataShape(0)
    val W = helper.dataShape(1)
    val H = helper.dataShape(2)

    val filter = helper.filter

    /**
     * chwFlag is to set image input of CHW or HWC
     * if true, the format is CHW
     * else, the format is HWC
     *
     * Note that currently CHW is commonly used
     * and HWC is often used in Tensorflow models
     */
    val chwFlag = if (modelType == "tensorflow") {
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
      InferenceSummary(".", helper.dateTime + "-ClusterServing"))

    val spark = helper.getSparkSession()

    logger.info(s"connected to redis " +
      s"${spark.conf.get("spark.redis.host")}:${spark.conf.get("spark.redis.port")}")



    var totalCnt: Int = 0
    var timeStamp: Int = 0

    // redis stream control
    val redisDB = new Jedis(helper.redisHost, helper.redisPort.toInt)
    val inputThreshold = 0.6 * 0.8
    val cutRatio = 0.5


    val serParams = new Params(helper.coreNum, helper.filter,
      chwFlag, helper.dataShape(0), helper.dataShape(1), helper.dataShape(2),
      _modelType = helper.modelType)

    val ssc = new StreamingContext(spark.sparkContext, new Duration(200))
    val acc = new LongAccumulator()
    helper.sc.register(acc)

    val receiver = new ServingReceiver()
    val images = ssc.receiverStream(receiver)

    //    val image = ssc.socketTextStream("localhost", 9999)

//    val images = ssc.createRedisXStream(Seq(ConsumerConfig("image_stream", "group1", "cli1",
//      batchSize = 512, block = 50)))

    images.foreachRDD{ m =>
      /**
       * This is reserved for future dynamic loading model
       */
      m.persist()

      if (!m.isEmpty) {
        val microBatchStart = System.nanoTime()
        val x = m.coalesce(1)
        acc.reset()

        val redisInfo = RedisUtils.getMapFromInfo(redisDB.info())

        if (redisInfo("used_memory").toLong >=
          redisInfo("maxmemory").toLong * inputThreshold) {
          redisDB.xtrim("image_stream",
            (redisDB.xlen("image_stream") * cutRatio).toLong, true)
        }
        /**
         * The streaming may be triggered somehow and it is possible
         * to get an empty batch
         *
         * If the batch is not empty, start preprocessing and predict here
         */


        val preProcessed = x.mapPartitions(it => {
          it.grouped(serParams.coreNum).flatMap(itemBatch => {
            itemBatch.indices.toParArray.map(i => {
              val uri = itemBatch(i)._1
              val tensor = PreProcessing(itemBatch(i)._2)
//              val uri = itemBatch(i).fields("uri")
//              val tensor = PreProcessing(itemBatch(i).fields("image"))
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
          InferenceStrategy(serParams, bcModel, acc, preProcessed, "single")
        } else {
          /**
           * In Normal mode, every model will use multiple thread to
           * achieve best latency. Thus, we only use a single model to
           * do sequential predict, maximizing the latency performance
           * and minimizing the memory usage.
           */
          InferenceStrategy(serParams, bcModel, acc, preProcessed, "all")
        }


        /**
         * Predict ends, start writing data to output queue
         */
        val resDf = spark.createDataFrame(postProcessed)

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
              .option("iterator.grouping.size", batchSize)
              .mode(SaveMode.Append).save()

            errFlag = false
          }

          catch {
            case e: redis.clients.jedis.exceptions.JedisDataException =>
              errFlag = true
              val errMsg = "not enough space in redis to write: " + e.toString
              logger.info(errMsg)
              println(errMsg)

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
              println(errMsg)

              Thread.sleep(3000)

          }
        }

        /**
         * Count the statistical data and write to summary
         */
        val microBatchEnd = System.nanoTime()
        println(s"Currently recs in redis: ${redisDB.xlen("image_stream")}")

        AsyncUtils.writeServingSummay(model, acc.value,
          microBatchStart, microBatchEnd, timeStamp, totalCnt)
          .onComplete{
            case Success(value) =>
              timeStamp += value._1
              totalCnt += value._2
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
    ssc.start()

    ClusterServingManager.listenTermination(helper, ssc)

    ssc.awaitTermination()

    assert(spark.streams.active.isEmpty)
    System.exit(0)
  }
}

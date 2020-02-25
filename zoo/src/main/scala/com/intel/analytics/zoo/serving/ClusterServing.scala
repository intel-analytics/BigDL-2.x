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
import com.intel.analytics.zoo.utils.ImageProcessing
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SaveMode}
import redis.clients.jedis.Jedis

import scala.concurrent.ExecutionContext.Implicits.global


object ClusterServing {
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
    val batchSize = helper.batchSize
    val topN = helper.topN
    val coreNum = helper.coreNum
    val nodeNum = helper.nodeNum
    val modelType = helper.modelType
    val blasFlag = helper.blasFlag

    val C = helper.dataShape(0)
    val W = helper.dataShape(1)
    val H = helper.dataShape(2)

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
    val inputThreshold = 0.6 * 0.8
    val cutRatio = 0.5

    val upperParNum = (coreNum - 1) / batchSize + 1

    val omp = if (sys.env.contains("OMP_NUM_THREADS")) {
      sys.env("OMP_NUM_THREADS")
    } else {
      1
    }

    logger.info(s"Core number is $coreNum, Using batchSize $batchSize, OMP $omp, model number $upperParNum")

    val query = images.writeStream.foreachBatch{ (batchDF: DataFrame, batchId: Long) =>
      val t1 = System.currentTimeMillis()
      logger.info(s"start time stamp $t1")
      /**
       * This is reserved for future dynamic loading model
       */
      val redisInfo = RedisUtils.getMapFromInfo(redisDB.info())

      if (redisInfo("used_memory").toLong >=
        redisInfo("maxmemory").toLong * inputThreshold) {
        redisDB.xtrim("image_stream",
          (redisDB.xlen("image_stream") * cutRatio).toLong, true)
      }

      batchDF.persist()
      val t2 = System.currentTimeMillis()
      logger.info(s"get to persist ${t2 - t1}")

//      val microBatchSize = batchDF.count()
      val t3 = System.currentTimeMillis()
//      logger.info(s"count ${t3 - t2}")
//      logger.info("Micro batch size " + microBatchSize.toString)


      if (!batchDF.isEmpty) {
        /**
         * The streaming may be triggered somehow and it is possible
         * to get an empty batch
         *
         * If the batch is not empty, start preprocessing and predict here
         */


        val microBatchStart = System.nanoTime()

        /**
         * Engine type controlling, for different engine type,
         * different partitioning and batching scheduling is used
         */
        val resultPartitions = if (blasFlag) {
          /**
           * In BLAS mode, every model could predict only using
           * a single thread, besides, batch size usually is not
           * over 64 in serving to achieve good latency. Thus, no
           * batching is required if the machine has over about 30 cores.           *
           */
          batchDF.rdd.mapPartitions(pathBytes => {
            pathBytes.grouped(coreNum).flatMap(pathBytesBatch => {
              pathBytesBatch.indices.toParArray.map(i => {
                val path = pathBytesBatch(i).getAs[String]("uri")
                val tensors = ImageProcessing.bytesToBGRTensor(java.util
                  .Base64.getDecoder.decode(pathBytesBatch(i).getAs[String]("image")))

                val localPartitionModel = bcModel.value
                val result = localPartitionModel.doPredict(tensors.addSingletonDimension()).toTensor

                val value = PostProcessing.getInfofromTensor(topN, result.squeeze())

                Record(path, value)
              })
            })
          })
        }

        else {
          /**
           * In Normal mode, every model will use multiple thread to
           * achieve best latency. Thus, we only use a single model to
           * do sequential predict, maximizing the latency performance
           * and minimizing the memory usage.
           */
          val pathBytesChunk = batchDF.rdd.mapPartitions(pathBytes => {
            pathBytes.grouped(coreNum).flatMap(pathBytesBatch => {
              pathBytesBatch.indices.toParArray.map(i => {
                val row = pathBytesBatch(i)
                val path = row.getAs[String]("uri")
                val tensors = ImageProcessing.bytesToBGRTensor(java.util
                  .Base64.getDecoder.decode(row.getAs[String]("image")), chwFlag)
                (path, tensors)

              })
            })
          })

          // per batch size should be equal to OMP_NUM_THEADS to get max performance

          val tensorArray = new Array[Tensor[Float]](upperParNum)


          (0 until upperParNum).foreach(i => {
            if (chwFlag) {
              tensorArray(i) = Tensor[Float](batchSize, C, H, W)
            } else {
              tensorArray(i) = Tensor[Float](batchSize, H, W, C)
            }
          })

          pathBytesChunk.mapPartitions(pathBytes => {
            pathBytes.grouped(coreNum).flatMap(batch => {

              val thisUpperParNum = (batch.size - 1) / batchSize + 1
              println(s"using $thisUpperParNum models to predict ${batch.size} records")
              (0 until thisUpperParNum).toParArray.flatMap(modelIndex => {
                val localModel = bcModel.value

                val beginIndex = modelIndex * batchSize
                val endIndex = if (beginIndex + batchSize < batch.size) {
                  beginIndex + batchSize
                } else {
                  batch.size
                }

                val thisTensor = if (beginIndex + batchSize < batch.size) {
                  tensorArray(modelIndex)
                } else {
                  if (chwFlag){
                    tensorArray(modelIndex).resize(endIndex - beginIndex, C, H, W)
                  } else {
                    tensorArray(modelIndex).resize(endIndex - beginIndex, H, W, C)
                  }

                }

                (beginIndex until endIndex).toParArray.foreach(nIndex => {
                  thisTensor.select(1, nIndex - beginIndex + 1).copy(batch(nIndex)._2)
                })
                if (modelType == "openvino") {
                  thisTensor.addSingletonDimension()
                }
                val result = localModel.doPredict(thisTensor).toTensor
                if (modelType == "openvino") {
//                  tensorArray(modelIndex).squeeze(1)
                  result.squeeze(1)
                }
                (beginIndex - beginIndex until endIndex - beginIndex).toParArray.map(i => {
                  val value = PostProcessing.getInfofromTensor(topN,
                    result.select(1, i + 1))
                  Record(batch(beginIndex + i)._1, value)
                })
              })
            })

          })
        }


        //   pathBytesChunk.mapPartitions(pathBytes => {
        //     val localModel = bcModel.value
        //     val t = if (chwFlag) {
        //       Tensor[Float](batchSize, C, H, W)
        //     } else {
        //       Tensor[Float](batchSize, H, W, C)
        //     }
        //     pathBytes.grouped(batchSize).flatMap(pathByteBatch => {
        //       val thisBatchSize = pathByteBatch.size

        //       (0 until thisBatchSize).toParArray
        //         .foreach(i => t.select(1, i + 1).copy(pathByteBatch(i)._2))

        //       val x = if (modelType == "openvino") {
        //         t.addSingletonDimension()
        //       } else {
        //         t
        //       }
        //       /**
        //        * addSingletonDimension method will modify the
        //        * original Tensor, thus if reuse of Tensor is needed,
        //        * have to squeeze it back.
        //        */
        //       val result = if (modelType == "openvino") {
        //         val res = localModel.doPredict(x).toTensor.squeeze()
        //         t.squeeze(1)
        //         res
        //       } else {
        //         localModel.doPredict(x).toTensor
        //       }

        //       (0 until thisBatchSize).toParArray.map(i => {
        //         val value = PostProcessing.getInfofromTensor(topN,
        //           result.select(1, i + 1).squeeze())
        //         Record(pathByteBatch(i)._1, value)
        //       })

        //     })
        //   })
        // }


        /**
         * Predict ends, start writing data to output queue
         */
        val resDf = spark.createDataFrame(resultPartitions)

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

        AsyncUtils.writeServingSummay(model, batchDF,
          microBatchStart, microBatchEnd, timeStamp, totalCnt)
          .onComplete(_ => None)
//        val microBatchLatency = (microBatchEnd - microBatchStart) / 1e9
//        val microBatchThroughPut = (microBatchSize / microBatchLatency).toFloat
//
//        totalCnt += microBatchSize.toInt
//        try {
//          if (model.inferenceSummary != null) {
//            (timeStamp until timeStamp + microBatchLatency.toInt).foreach( time => {
//              model.inferenceSummary.addScalar(
//                "Serving Throughput", microBatchThroughPut, time)
//            })
//
//            model.inferenceSummary.addScalar(
//              "Total Records Number", totalCnt, batchId)
//          }
//        }
//        catch {
//          case e: Exception =>
//            /**
//             * If been interrupted by stop signal, do nothing
//             * End the streaming until this micro batch process ends
//             */
//            logger.info("Summary not supported. skipped.")
//        }

//
//        timeStamp += microBatchLatency.toInt
//
//        logger.info(microBatchSize +
//          " inputs predict ended, time elapsed " + microBatchLatency.toString)
        val t4 = System.currentTimeMillis()
        logger.info(s"total  ${t4 - t1}")
        logger.info(s"end time stamp $t4")
      }
    }

    /**
     * Start the streaming, and listen to stop signal
     * The stop signal will be listened by another thread
     * with interval of 1 second.
     */
    val servingQuery = query.start()

    ClusterServingManager.listenTermination(helper, servingQuery)

    servingQuery.awaitTermination()

    assert(spark.streams.active.isEmpty)
    System.exit(0)
  }
}


package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.inference.{InferenceModel, InferenceSummary}
import com.intel.analytics.zoo.serving.ClusterServing.Record
import com.intel.analytics.zoo.serving.utils.ClusterServingHelper
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.streaming.{Duration, StreamingContext}
import com.redislabs.provider.redis.streaming._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.broadcast.Broadcast

object SSServing {
  Logger.getLogger("org").setLevel(Level.ERROR)
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
    val ssc = new StreamingContext(spark.sparkContext, new Duration(100))

//    val image = ssc.socketTextStream("localhost", 9999)
    val image = ssc.createRedisXStream(Seq(ConsumerConfig("image_stream", "group1", "cli1")))
    image.foreachRDD(x => {

      x.persist()
      if (!x.isEmpty()) {
        println(s"time 1 ${System.currentTimeMillis().toString}")
        val pathBytesChunk = x.mapPartitions(it => {
          it.grouped(coreNum).flatMap(itemBatch => {
            itemBatch.indices.toParArray.map(i => {

              val uri = itemBatch(i).fields("uri")
              val tensor = PreProcessing(itemBatch(i).fields("image"))
              (uri, tensor)
            })
          })
        })
        val resRDD = pathBytesChunk.mapPartitions(pathBytes => {
          val localModel = bcModel.value
          val t = if (chwFlag) {
            Tensor[Float](batchSize, C, H, W)
          } else {
            Tensor[Float](batchSize, H, W, C)
          }
          pathBytes.grouped(batchSize).flatMap(pathByteBatch => {
            val thisBatchSize = pathByteBatch.size

            (0 until thisBatchSize).toParArray
              .foreach(i => t.select(1, i + 1).copy(pathByteBatch(i)._2))

            val thisTensor = if (chwFlag){
              t.resize(thisBatchSize, C, H, W)
            } else {
              t.resize(thisBatchSize, H, W, C)
            }

            val x = if (modelType == "openvino") {
              thisTensor.addSingletonDimension()
            } else {
              thisTensor
            }
            /**
             * addSingletonDimension method will modify the
             * original Tensor, thus if reuse of Tensor is needed,
             * have to squeeze it back.
             */

            val result = if (modelType == "openvino") {
              val res = localModel.doPredict(x).toTensor[Float].squeeze()
              t.squeeze(1)
              res
            } else {
              localModel.doPredict(x).toTensor[Float]
            }

            (0 until thisBatchSize).toParArray.map(i => {
              val value = PostProcessing(result.select(1, i + 1).squeeze())
              Record(pathByteBatch(i)._1, value)
            })

          })

        })
        val resDf = spark.createDataFrame(resRDD)
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

        println(s"time 2 ${System.currentTimeMillis().toString}")
      }



    })
    ssc.start()
    ssc.awaitTermination()
  }

}

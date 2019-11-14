package com.intel.analytics.zoo.serving

import java.util.Base64

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.feature.image.OpenCVMethod
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import com.intel.analytics.zoo.pipeline.inference.{InferenceModel, InferenceSupportive}
import com.intel.analytics.zoo.serving.ClusterServing.{getClass, logger}
import com.intel.analytics.zoo.serving.utils.ClusterServingHelper
import com.intel.analytics.zoo.utils.ImageProcessing
import org.apache.commons.lang.ObjectUtils
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.storage.StorageLevel.MEMORY_ONLY
import org.opencv.imgcodecs.Imgcodecs

object ClusterServingApp extends InferenceSupportive {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger
    .getLogger("com.intel.analytics.zoo.feature.image")
    .setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger: Logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.engineType", "mkldnn")

    val helper = new ClusterServingHelper()
    helper.initArgs(args)
    System.setProperty("bigdl.localMode", "false")
    System.setProperty("bigdl.coreNumber", helper.nodeNum.toString)

    helper.initContext()
    val coreNumber = EngineRef.getCoreNumber()
    val nodeNum = EngineRef.getNodeNumber()
    val eType = EngineRef.getEngineType()
    logger.info("Engine Type is " + eType)
    logger.info("Core number is running at " + coreNumber.toString)
    logger.info("Node number is running at " + nodeNum.toString)
    logger.info("stream.read.batch.size is running at " + helper.batchSize)

    val spark = helper.getSparkSession()
    val model = new InferenceModel(1)
    model.doLoad(helper.weightPath)
    val broadcasted = spark.sparkContext.broadcast(model)

    logger.info(s"connected to redis " +
      s"${spark.conf.get("spark.redis.host")}:${spark.conf.get("spark.redis.port")}")

    val batchSize = helper.batchSize
    val topN = helper.topN
    val images = spark.readStream
      .format("redis")
      .option("stream.keys", "image_stream")
      .option("stream.read.batch.size", 500)
      .option("stream.read.block", 500)
      .option("stream.parallelism", EngineRef.getNodeNumber())
      .schema(
        StructType(
          Array(
            StructField("id", StringType),
            StructField("path", StringType),
            StructField("image", StringType)
          )))
      .load()

    val query = images.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        {
          batchDF.persist(MEMORY_ONLY)
          val microBatchSize = batchDF.count()
          logger.info(s"Get batch $batchId, batchSize: $microBatchSize")
          println(s"partitions: ${batchDF.rdd.partitions.size}")
          batchDF.rdd.foreachPartition(partition => {
            println(s"------------ ${partition.size}")
          })
          val results = batchDF.rdd.map(image => {
            val localModel = broadcasted.value
            println(s"################### ${ObjectUtils.identityToString(localModel)}")
            val bytes = Base64.getDecoder.decode(image.getAs[String]("image"))
            val tensor = timing("preprocess single one") {
              ImageProcessing.bytesToBGRTensor(bytes)
            }
            val result = timing("predict single one") {
              localModel.doPredict(tensor)
            }
            result
          }).collect()
          println(s"================ ${results}")
          batchDF.unpersist()
        }
      }
      .start()
    query.awaitTermination()
  }

}

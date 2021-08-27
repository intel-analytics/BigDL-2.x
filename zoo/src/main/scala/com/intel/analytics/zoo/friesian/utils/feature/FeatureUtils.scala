package com.intel.analytics.zoo.friesian.utils.feature

import java.util.{Base64, List => JList, Map => JMap}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.friesian.utils.EncodeUtils.objToBytes
import com.intel.analytics.zoo.friesian.generated.feature.{Features, IDs}
import com.intel.analytics.zoo.friesian.utils.{EncodeUtils, gRPCHelper}
import io.grpc.StatusRuntimeException
import org.apache.spark.sql.functions.{col, max, udf}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

object FeatureUtils {
  var helper: gRPCHelper = _
  val logger: Logger = Logger.getLogger(getClass)

  def loadUserItemFeaturesRDD(spark: SparkSession): Unit = {
    assert(FeatureUtils.helper.initialUserDataPath != null ||
      FeatureUtils.helper.initialItemDataPath != null, "initialUserDataPath or " +
      "initialItemDataPath should be provided if loadInitialData is true")
    if (FeatureUtils.helper.initialUserDataPath != null) {
      assert(FeatureUtils.helper.userIDColumn != null)
      assert(FeatureUtils.helper.userFeatureColArr != null)
      val userFeatureColumns = FeatureUtils.helper.userIDColumn +: FeatureUtils.helper.userFeatureColArr
      var userFeatures = spark.read.parquet(FeatureUtils.helper.initialUserDataPath)
      userFeatures = userFeatures.select(userFeatureColumns.map(col):_*).distinct()
      val usercnt = userFeatures.select(FeatureUtils.helper.userIDColumn).distinct().count()
      logger.info(s"Load ${usercnt} user features into redis.")
      val userCols = userFeatures.columns
      if (usercnt >= 2000000) {
        userFeatures = userFeatures.repartition(200)
      }
      val userFeatureRDD = userFeatures.rdd.map(row => {
        encodeRowWithCols(row, userCols)
      })
      logger.info(s"UserFeatureRDD partition number: ${userFeatureRDD.getNumPartitions}")
      userFeatureRDD.foreachPartition { partition =>
        val redis = RedisUtils.getInstance()
        redis.Hset("userid", partition.toArray)
      }
    }

    if (FeatureUtils.helper.initialItemDataPath != null) {
      assert(FeatureUtils.helper.itemIDColumn != null)
      assert(FeatureUtils.helper.itemFeatureColArr != null)
      val itemFeatureColumns = FeatureUtils.helper.itemIDColumn +: FeatureUtils.helper.itemFeatureColArr
      var itemFeatures = spark.read.parquet(FeatureUtils.helper.initialItemDataPath)
      itemFeatures = itemFeatures.select(itemFeatureColumns.map(col):_*)
      val itemcnt = itemFeatures.select(FeatureUtils.helper.itemIDColumn).distinct().count()
      logger.info(s"Load ${itemcnt} item features into redis.")
      val itemCols = itemFeatures.columns
      if (itemcnt >= 2000000) {
        itemFeatures = itemFeatures.repartition(200)
      }
      val itemFeatureRDD = itemFeatures.rdd.map(row => {
        encodeRowWithCols(row, itemCols)
      })
      logger.info(s"ItemFeatureRDD partition number: ${itemFeatureRDD.getNumPartitions}")
      itemFeatureRDD.foreachPartition { partition =>
        val redis = RedisUtils.getInstance()
        redis.Hset("itemid", partition.toArray)
      }
    }
    logger.info(s"Insert finished")
  }

  def encodeRow(row: Row): JList[String] = {
    val rowSeq = row.toSeq
    val id = rowSeq.head.toString
    val encodedValue = java.util.Base64.getEncoder.encodeToString(objToBytes(rowSeq))
    List(id, encodedValue).asJava
  }

  def encodeRowWithCols(row: Row, cols: Array[String]): JList[String] = {
    val rowSeq = row.toSeq
    val id = rowSeq.head.toString
    val colValueMap = (cols zip rowSeq).toMap
    val encodedValue = java.util.Base64.getEncoder.encodeToString(
      objToBytes(colValueMap))
    List(id, encodedValue).asJava
  }

  def loadUserData(dataDir: String, userIdCol: String): Array[Int] = {
    val spark = SparkSession.builder.getOrCreate
    val df = spark.read.parquet(dataDir)
    df.select(userIdCol).distinct.limit(1000).rdd.map(row => {
//      row.getLong(0).toInt
      row.getInt(0)
    }).collect
  }

  def doPredict(ids: IDs, model: InferenceModel): JList[String] = {
    val idsScala = ids.getIDList.asScala
    val input = Tensor[Float](T.seq(idsScala))
    val result: Tensor[Float] = model.doPredict(input).toTensor
    idsScala.indices.map(idx => {
      val dTensor: Tensor[Float] = result.select(1, idx + 1)
      java.util.Base64.getEncoder.encodeToString(
        objToBytes(dTensor))
    }).toList.asJava
  }

  def predictFeatures(features: Features, model: InferenceModel, featureColumns: Array[String],
                      idColumn: String): JList[String] = {
    val featureArr = FeatureUtils.featuresToObject(features)
      .filter(idx => idx != null)
    if (featureArr.length == 0) {
      throw new Exception("Cannot find target user/item in redis.")
    }
    val featureListIDArr = featureArr.map(f => {
      val fMap = f.asInstanceOf[Map[String, AnyRef]]
      val fList = featureColumns.map(colName => {
        fMap.getOrElse(colName, -1)
      }) :+ fMap.getOrElse(idColumn, -1)
      fList
    })

    val tensorArray = ArrayBuffer[Tensor[Float]]()

    featureColumns.indices.foreach(idx => {
      var singleDim = true
      val converted = featureListIDArr.map(singleFeature => {
        // TODO: null
        singleFeature(idx) match {
          case d: Int => Array(d.toFloat)
          case d: Float => Array(d)
          case d: Long => Array(d.toFloat)
          case d: mutable.WrappedArray[AnyRef] =>
            singleDim = false
            var isNumber = true
            if (d.nonEmpty) {
              if (d(0).isInstanceOf[String]) {
                isNumber = false
              }
            }
            if (isNumber) {
              d.toArray.map(_.asInstanceOf[Number].floatValue())
            } else {
              d.toArray.map(_.asInstanceOf[String].toFloat)
            }
          case _ => throw new IllegalArgumentException("")
        }
      })
      val inputTensor = if (singleDim) {
        Tensor[Float](converted.flatten, Array(converted.length))
      } else {
        Tensor[Float](converted.flatten, Array(converted.length, converted(0).length))
      }
      tensorArray.append(inputTensor)
    })
    val inputFeature = T.array(tensorArray.toArray)
    val result: Tensor[Float] = model.doPredict(inputFeature).toTensor
    val ids = featureListIDArr.map(_.last.asInstanceOf[Int])
    ids.indices.map(idx => {
      val dTensor: Tensor[Float] = result.select(1, idx + 1)
      java.util.Base64.getEncoder.encodeToString(
        objToBytes(dTensor))
    }).toList.asJava
  }

  def featuresToObject(features: Features): Array[AnyRef] = {
    val b64Features = features.getB64FeatureList.asScala
    b64Features.map(feature => {
      if (feature == "") {
        null
      } else {
        EncodeUtils.bytesToObj(Base64.getDecoder.decode(feature))
      }
    }).toArray
  }
}

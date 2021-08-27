package com.intel.analytics.zoo.friesian.utils.recommend

import java.util.Base64
import java.util

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.friesian.generated.azinference.{AZInferenceGrpc, Content, Prediction}
import com.intel.analytics.zoo.friesian.generated.feature.Features
import com.intel.analytics.zoo.friesian.utils.{EncodeUtils, gRPCHelper}
import com.intel.analytics.zoo.friesian.utils.feature.FeatureUtils
import io.grpc.StatusRuntimeException
import org.apache.log4j.Logger
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import com.intel.analytics.zoo.friesian.utils.gRPCHelper

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

object RecommendUtils {
  var helper: gRPCHelper = _
  val logger: Logger = Logger.getLogger(getClass)

  def featuresToRankingInputSet(userFeatures: Features, itemFeatures: Features, batchSize: Int)
  : (Array[Int], Array[Table]) = {
    val userFeatureArr = FeatureUtils.featuresToObject(userFeatures)
    assert(userFeatureArr.length == 1, "userFeatures length should be 1")
    val userFeature = userFeatureArr(0).asInstanceOf[Map[String, AnyRef]]
    // TODO: not found update
    val itemFeatureArr = FeatureUtils.featuresToObject(itemFeatures)
      .filter(idx => idx != null)
    logger.info("Got item feature: " + itemFeatureArr.length)

    val batchSizeUse = if (batchSize <= 0) {
      itemFeatureArr.length
    } else {
      batchSize
    }
    if (batchSizeUse == 0) {
      throw new Exception("The recommend service got 0 valid item features. Please make sure " +
        "your initial datasets are matched.")
    }
    val inferenceColumns = RecommendUtils.helper.inferenceColArr

    val userItemFeatureItemIdArr = itemFeatureArr.map(itemF => {
      val itemFMap = itemF.asInstanceOf[Map[String, AnyRef]]
      val userItemFMap = itemFMap.++(userFeature)
      val featureList = inferenceColumns.map(colName => {
        userItemFMap.getOrElse(colName, -1)
      })
      val itemId = userItemFMap.getOrElse(RecommendUtils.helper.itemIDColumn, -1).asInstanceOf[Int]
      (itemId, featureList)
    })
    val itemIDArr = userItemFeatureItemIdArr.map(_._1)
    val userItemFeatureArr = userItemFeatureItemIdArr.map(_._2)
    val batchedFeatureArr = userItemFeatureArr.sliding(batchSizeUse, batchSizeUse).toArray
    val batchedActivityList = batchedFeatureArr.map(featureArr => {
      val tensorArray = ArrayBuffer[Tensor[Float]]()
      inferenceColumns.indices.foreach(idx => {
        var singleDim = true
        val converted = featureArr.map(singleFeature => {
          // TODO: null
          singleFeature(idx) match {
            case d: Int => Array(d.toFloat)
            case d: Float => Array(d)
            case d: Long => Array(d.toFloat)
            case d: mutable.WrappedArray[AnyRef] =>
              singleDim = false
              d.toArray.map(_.asInstanceOf[Number].floatValue())
            case d => throw new IllegalArgumentException(s"Illegal input: ${d}")
          }
        })
        val inputTensor = if (singleDim) {
          Tensor[Float](converted.flatten, Array(converted.length))
        } else {
          // TODO: empty
          Tensor[Float](converted.flatten, Array(converted.length, converted(0).length))
        }
        tensorArray.append(inputTensor)
      })
      T.array(tensorArray.toArray)
    })

    (itemIDArr, batchedActivityList)
  }

  def doPredictParallel(inputArr: Array[Activity],
                        inferenceStub: AZInferenceGrpc.AZInferenceBlockingStub): Array[String] = {
    val resultArr = inputArr.indices.toParArray.map(idx => {
      val input = Base64.getEncoder.encodeToString(EncodeUtils.objToBytes(inputArr(idx)))
      val predContent = Content.newBuilder().setEncodedStr(input).build()
      var result: Prediction = null
      try {
          result = inferenceStub.doPredict(predContent)
      } catch {
        case e: StatusRuntimeException => throw e
      }

      result.getPredictStr
    })
    resultArr.toArray
  }

  def getTopK(result: Array[String], itemIDArr: Array[Int], k: Int): (Array[Int], Array[Float]) = {
    val resultArr = result.indices.toParArray.map(idx => {
      val resultStr = result(idx)
      val resultStrArr = resultStr.replaceAll("\\[", "").dropRight(2).split("\\],")
      resultStrArr.map(a => {
        a.split(",")(0).toFloat
      })
    }).toArray
    val flattenResult = resultArr.flatten
    val zipped = itemIDArr zip flattenResult
    val sorted = zipped.sortWith(_._2 > _._2).take(k)
    val sortedId = sorted.map(_._1)
    val sortedProb = sorted.map(_._2)
    (sortedId, sortedProb)
  }

  // For wnd validation
  def loadResultParquet(resultPath: String):
  (util.Map[Integer, Integer], util.Map[Integer, java.lang.Float]) = {
    val spark = SparkSession.builder.getOrCreate
    val df = spark.read.parquet(resultPath)
    val userItemMap = collection.mutable.Map[Int, Int]()
    val userPredMap = collection.mutable.Map[Int, Float]()
    df.collect().foreach(row => {
      val userId = row.getInt(0)
      val itemId = row.getInt(1)
      val pred = row.getAs[DenseVector](2).toArray(0).toFloat
      userItemMap.update(userId, itemId)
      userPredMap.update(userId, pred)
    })
    (userItemMap.asJava.asInstanceOf[util.Map[Integer, Integer]], userPredMap.asJava
      .asInstanceOf[util.Map[Integer, java.lang.Float]])
  }
}

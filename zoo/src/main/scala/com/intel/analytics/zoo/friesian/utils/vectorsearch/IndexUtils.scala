package com.intel.analytics.zoo.friesian.utils.vectorsearch

import java.io.File

import org.apache.log4j.Logger
import java.util.{List => JList}

import com.codahale.metrics.Timer
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.faiss.swighnswlib.floatArray
import com.intel.analytics.zoo.friesian.service.indexing.IndexService
import com.intel.analytics.zoo.friesian.utils.gRPCHelper
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{array, col, udf}
import org.spark_project.jetty.server.Authentication.Wrapped
import com.intel.analytics.zoo.friesian.utils.Utils

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


object IndexUtils {
  var helper: gRPCHelper = _
  private val logger = Logger.getLogger(classOf[IndexService].getName)

  def loadItemData(indexService: IndexService, dataDir: String, model:InferenceModel,
                   batchSize: Int = 0): Unit = {
    val spark = SparkSession.builder.getOrCreate
    assert(IndexUtils.helper.itemIDColumn != null, "itemIdColumn should be provided if " +
      "loadSavedIndex=false")
    if (model != null) {
      var df = spark.read.parquet(dataDir)
      val cnt = df.select(helper.getItemIDColumn).distinct().count()
      logger.info("Total id count: " + cnt)
      val partitionNum: Int = (cnt.toFloat/batchSize).ceil.toInt
      df = df.repartition(partitionNum)
      assert(IndexUtils.helper.itemFeatureColArr != null, "itemFeatureColumns should be provided " +
        "if loadSavedIndex=false and itemModelPath != null")
      val itemFeatureColumns = IndexUtils.helper.itemIDColumn +: IndexUtils.helper.itemFeatureColArr
      df = df.select(itemFeatureColumns.map(col):_*).distinct()
      val result = df.rdd.mapPartitions(rows => {
        val rowArray = rows.toArray
        val idList = rowArray.map(row => {
          row.getInt(0)
        })
        val tensorArray = ArrayBuffer[Tensor[Float]]()
        IndexUtils.helper.itemFeatureColArr.indices.foreach(idx => {
          var singleDim = true
          val converted = rowArray.map(singleFeature => {
            // TODO: null
            singleFeature(idx + 1) match {
              case d: Int => Array(d.toFloat)
              case d: Float => Array(d)
              case d: Long => Array(d.toFloat)
              case d: mutable.WrappedArray[AnyRef] =>
                singleDim = false
                d.toArray.map(_.asInstanceOf[Number].floatValue())
              case _ => throw new IllegalArgumentException("")
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
        val inputFeature = T.array(tensorArray.toArray)
        val result: Tensor[Float] = IndexUtils.helper.itemModel.doPredict(inputFeature).toTensor
        val resultFlattenArr = result.storage().array()
        Array((resultFlattenArr, idList)).iterator
      }).collect()
      result.foreach(partResult => {
        val resultFlattenArr = partResult._1
        val idList = partResult._2
        indexService.addWithIds(resultFlattenArr, idList)
      })
    } else {
      val itemFeatureColumns = Array(IndexUtils.helper.itemIDColumn, "prediction")
      val parquetList = getListOfFiles(dataDir)
      logger.info(s"parquetList length: ${parquetList.length}")
      val readList = parquetList.sliding(10, 10).toArray
      val start = System.currentTimeMillis()
      for (parquetFiles <- readList) {
        var df = spark.read.parquet(parquetFiles: _*)
        df = df.select(itemFeatureColumns.map(col): _*).distinct()
        val data = df.rdd.map(row => {
          val id = row.getInt(0)
          val data = row.getAs[DenseVector](1).toArray.map(_.toFloat)
          (id, data)
        }).collect()
        val resultFlattenArr = data.flatMap(_._2)
        val idList = data.map(_._1)
        indexService.addWithIds(resultFlattenArr, idList)
      }
      val end = System.currentTimeMillis()
      logger.info(s"Building index takes: ${(end - start) / 1000}s")
      indexService.save("./2tower_item_full.idx")
    }
  }

  def constructActivity(data: JList[Any]): Tensor[Float] = {
    Tensor[Float](T.seq(data.asScala.map {
      case d: Int => d.toFloat
      case d: Double => d.toFloat
      case d: Float => d
      case d  =>
        throw new IllegalArgumentException(s"Only numeric values are supported, but got ${d}")
    }))
  }

  def activityToFloatArr(data: Activity): Array[Float] = {
    val dTensor: Tensor[Float] = data.toTensor
    val result = dTensor.squeeze(1).toArray()
    result
  }

  def getListOfFiles(dir: String): List[String] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      logger.info("file exists & dir")
      d.listFiles.filter(_.isFile).toList.map(_.getAbsolutePath).filter(!_.endsWith("SUCCESS"))
    } else {
      logger.info(s"empty, exists: ${d.exists()}, dir: ${d.isDirectory}")
      List[String]()
    }
  }
}

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

package com.intel.analytics.zoo.friesian.python

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.friesian.feature.Utils
import java.util.{List => JList}

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{array, broadcast, col, log => sqllog, row_number,
  spark_partition_id, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.math.{log => mathlog}

object PythonFriesian {
  def ofFloat(): PythonFriesian[Float] = new PythonFriesian[Float]()

  def ofDouble(): PythonFriesian[Double] = new PythonFriesian[Double]()
}

class PythonFriesian[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  val numericTypes: List[String] = List("long", "double", "integer")

  def dlrmPreprocessRDD(paths: JList[String], CATColumns: JList[String], INTColumns: JList[String],
                     frequencyLimit: String = null): Unit = {
//    val CATCols = CATColumns.asScala
//    val INTCols = INTColumns.asScala
    val start = System.nanoTime()
    val df = readParquet(paths)
    val dfStringIdxList = assignStringIdx2(df, CATColumns, frequencyLimit)
    var allDataDf = readParquet(paths.subList(0, paths.size() - 1))
    for (i <- 0 until CATColumns.size()) {
      val colName = CATColumns.get(i)
      val model = dfStringIdxList.get(i)
      // missing check would_broadcast
      val broadcastModel = broadcast(model)
      allDataDf = allDataDf
        .join(broadcastModel, allDataDf(colName) === broadcastModel(colName), joinType="left")
        .drop(colName)
        .withColumnRenamed("id", colName)
    }
    val preprocessed = allDataDf.rdd.map(row => {
      val intFeatures = new ArrayBuffer[Float]()
      for( i <- 1 to 13) {
        if (row.isNullAt(i)) {
          intFeatures.append(0)
        }
        else {
          val intFeature = row.getInt(i)
          if (intFeature < 0) {
            intFeatures.append(0)
          } else {
            intFeatures.append(mathlog(intFeature + 1).toFloat)
          }
        }
      }
      val catFeatures = new ArrayBuffer[Int]()
      for( i <- 14 to 39) {
        if (row.isNullAt(i)) {
          catFeatures.append(0)
        }
        else {
          catFeatures.append(row.getInt(i))
        }
      }
      // (y, X_int, X_cat)
      (row.getInt(0), intFeatures.toList.asJava, catFeatures.toList.asJava)
    })
    preprocessed.count()
    val end = System.nanoTime
    println("scala process time: " + (end - start) / 1e9d)
  }

  def dlrmPreprocessReturnDF(paths: JList[String], CATColumns: JList[String],
                             INTColumns: JList[String],
                             frequencyLimit: String = null): DataFrame = {
    val CATCols = CATColumns.asScala
    val INTCols = INTColumns.asScala
    val start = System.nanoTime()
    val df = readParquet(paths)
    val dfStringIdxList = assignStringIdx2(df, CATColumns, frequencyLimit)
    var allDataDf = readParquet(paths.subList(0, paths.size() - 1))
    for (i <- 0 until CATColumns.size()) {
      val colName = CATColumns.get(i)
      val model = dfStringIdxList.get(i)
      // missing check would_broadcast
      val broadcastModel = broadcast(model)
      allDataDf = allDataDf
        .join(broadcastModel, allDataDf(colName) === broadcastModel(colName), joinType="left")
        .drop(colName)
        .withColumnRenamed("id", colName)
    }
    val allColumns = INTCols ++ CATCols
    allDataDf = fillNaInt(allDataDf, 0, allColumns.asJava)
    val zeroThreshold = (value: Int) => {
      if (value < 0) 0 else value
    }

    val zeroThresholdUDF = udf(zeroThreshold)
    for(colName <- INTCols) {
      allDataDf = allDataDf.withColumn(colName, sqllog(zeroThresholdUDF(col(colName)) + 1))
    }
    allDataDf = allDataDf.withColumn("X_int", array(INTCols.map(x => col(x)):_*))
    allDataDf = allDataDf.withColumn("X_cat", array(CATCols.map(x => col(x)):_*))
    allDataDf = allDataDf.select(col("_c0"), col("X_int"), col("X_cat"))
    val end = System.nanoTime
    println("scala process time: " + (end - start) / 1e9d)
    allDataDf
  }

  def dlrmPreprocess(paths: JList[String], CATColumns: JList[String], INTColumns: JList[String],
                     frequencyLimit: String = null): Unit = {
    val start = System.nanoTime()
    val allDataDf = dlrmPreprocessReturnDF(paths, CATColumns, INTColumns, frequencyLimit)
    allDataDf.rdd.count()
    val end = System.nanoTime
    println("scala process time2: " + (end - start) / 1e9d)
  }

  def dlrmPreprocessReturnDFCompute(paths: JList[String], CATColumns: JList[String],
                                    INTColumns: JList[String], frequencyLimit: String = null)
  : DataFrame = {
    val start = System.nanoTime()
    val allDataDf = dlrmPreprocessReturnDF(paths, CATColumns, INTColumns, frequencyLimit)
    allDataDf.rdd.count()
    val end = System.nanoTime
    println("scala process time2: " + (end - start) / 1e9d)
    allDataDf
  }

  def readParquet(paths: JList[String]): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    val pathsScala = paths.asScala
    spark.read.parquet(pathsScala: _*)
  }

  def fillNa(df: DataFrame, fillVal: Any = 0, columns: JList[String] = null): DataFrame = {
    val cols = if (columns == null) {
      df.columns
    } else {
      columns.asScala.toArray
    }

    val cols_idx = cols.map(col_n => {
      val idx = df.columns.indexOf(col_n)
      if(idx == -1) {
        throw new IllegalArgumentException(s"The column name ${col_n} does not exist")
      }
      idx
    })

    Utils.fillNaIndex(df, fillVal, cols_idx)
  }

  def fillNaInt(df: DataFrame, fillVal: Int = 0, columns: JList[String] = null): DataFrame = {
    val cols = if (columns == null) {
      df.columns
    } else {
      columns.asScala.toArray
    }

    val schema = df.schema

    val cols_idx = cols.map(col_n => {
      val idx = df.columns.indexOf(col_n)
      if(idx == -1) {
        throw new IllegalArgumentException(s"The column name ${col_n} does not exist")
      }
      if (schema(idx).dataType.typeName != "integer") {
        throw new IllegalArgumentException(s"Only columns of IntegerType are supported")
      }
      idx
    })

    val dfUpdated = df.rdd.map(row => {
      val origin = row.toSeq.toArray
      for (idx <- cols_idx) {
        if (row.isNullAt(idx)) {
          origin.update(idx, fillVal)
        }
      }
      Row.fromSeq(origin)
    })

    val spark = df.sparkSession
    spark.createDataFrame(dfUpdated, schema)
  }

  def assignStringIdx(df_list: JList[DataFrame]): JList[DataFrame] = {
    val idx_df_list = (0 until df_list.size).map(x => {
      val df = df_list.get(x)
      df.cache()
      val count_list: Array[(Int, Int)] = df.rdd.mapPartitions(Utils.getPartitionSize).collect()
      val base_dict = scala.collection.mutable.Map[Int, Int]()
      var running_sum = 0
      for (count_tuple <- count_list) {
        base_dict += (count_tuple._1 -> running_sum)
        running_sum += count_tuple._2
      }
      val base_dict_bc = df.rdd.sparkContext.broadcast(base_dict)
      val df_with_part_id = df.withColumn("part_id", spark_partition_id())
      val windowSpec = Window.partitionBy("part_id").orderBy("count")
      val df_row_number = df_with_part_id.withColumn("row_number", row_number.over(windowSpec))
      val get_label = udf((part_id: Int, row_number: Int) => {
        row_number + base_dict_bc.value.getOrElse(part_id, 0)
      })
      val df_string_idx = df_row_number
        .withColumn("id", get_label(col("part_id"), col("row_number")))
        .drop("part_id", "row_number", "count")
//      df.unpersist()
      df_string_idx
    })
    idx_df_list.asJava
  }

  def assignStringIdx2(df: DataFrame, columns: JList[String], frequencyLimit: String = null)
  : JList[DataFrame] = {
    val freq_list = frequencyLimit.split(",")
    var default_limit: Option[Int] = None
    val freq_map = scala.collection.mutable.Map[String, Int]()
    for (fl <- freq_list) {
      val frequency_pair = fl.split(":")
      if (frequency_pair.length == 1) {
        default_limit = Some(frequency_pair(0).toInt)
      } else if (frequency_pair.length == 2) {
        freq_map += (frequency_pair(0) -> frequency_pair(1).toInt)
      }
    }
    val cols = columns.asScala.toList
    cols.map(col_n => {
      val df_col = df
        .select(col_n)
        .filter(s"${col_n} is not null")
        .groupBy(col_n)
        .count()
      val df_col_filtered = if (freq_map.contains(col_n)) {
        df_col.filter(s"count >= ${freq_map(col_n)}")
      } else if (default_limit.isDefined) {
        df_col.filter(s"count >= ${default_limit.get}")
      } else {
        df_col
      }

//      df_col_filtered.cache()
      val df_with_part_id = df_col_filtered.withColumn("part_id", spark_partition_id())
      df_with_part_id.cache()
      val count_list: Array[(Int, Int)] = df_with_part_id.rdd.mapPartitions(Utils.getPartitionSize)
        .collect()
      val base_dict = scala.collection.mutable.Map[Int, Int]()
      var running_sum = 0
      for (count_tuple <- count_list) {
        base_dict += (count_tuple._1 -> running_sum)
        running_sum += count_tuple._2
      }
      val base_dict_bc = df_col_filtered.rdd.sparkContext.broadcast(base_dict)

      val windowSpec = Window.partitionBy("part_id").orderBy("count")
      val df_row_number = df_with_part_id.withColumn("row_number", row_number.over(windowSpec))
      val get_label = udf((part_id: Int, row_number: Int) => {
        row_number + base_dict_bc.value.getOrElse(part_id, 0)
      })
      df_row_number
        .withColumn("id", get_label(col("part_id"), col("row_number")))
        .drop("part_id", "row_number", "count")
    }).asJava
  }

  def compute(df: DataFrame): Unit = {
    df.rdd.count()
  }

  def log(df: DataFrame, columns: JList[String]): DataFrame = {
    var resultDF = df
    val zeroThreshold = (value: Int) => {
      if (value < 0) 0 else value
    }

    val zeroThresholdUDF = udf(zeroThreshold)
    for(i <- 0 to columns.size()) {
      val colName = columns.get(i)
      resultDF = resultDF.withColumn(colName, sqllog(zeroThresholdUDF(col(colName)) + 1))
    }
    resultDF
  }
}

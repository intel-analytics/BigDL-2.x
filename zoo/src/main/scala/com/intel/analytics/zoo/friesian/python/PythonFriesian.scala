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
import org.apache.spark.sql.DataFrame
import java.util.{List => JList}

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number, spark_partition_id, udf}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object PythonFriesian {
  def ofFloat(): PythonFriesian[Float] = new PythonFriesian[Float]()

  def ofDouble(): PythonFriesian[Double] = new PythonFriesian[Double]()
}

class PythonFriesian[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  val numericTypes: List[String] = List("long", "double", "integer")

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

      df_col_filtered.cache()
      val count_list: Array[(Int, Int)] = df_col_filtered.rdd.mapPartitions(Utils.getPartitionSize)
        .collect()
      val base_dict = scala.collection.mutable.Map[Int, Int]()
      var running_sum = 0
      for (count_tuple <- count_list) {
        base_dict += (count_tuple._1 -> running_sum)
        running_sum += count_tuple._2
      }
      val base_dict_bc = df_col_filtered.rdd.sparkContext.broadcast(base_dict)
      val df_with_part_id = df_col_filtered.withColumn("part_id", spark_partition_id())
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
}

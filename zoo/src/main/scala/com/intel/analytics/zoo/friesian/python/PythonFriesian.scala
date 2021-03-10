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
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession, internal}
import java.util.{List => JList, Map => JMap}

import org.apache.spark.TaskContext
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{broadcast, col, count, max, min, row_number, spark_partition_id, udf}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object PythonFriesian {
  def ofFloat(): PythonFriesian[Float] = new PythonFriesian[Float]()

  def ofDouble(): PythonFriesian[Double] = new PythonFriesian[Double]()
}

class PythonFriesian[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def fillNA(df: DataFrame, fillVal: Int = 0, columns: JList[String] = null): DataFrame = {
    val schema = df.schema
    val dfUpdated = df.rdd.map(row => {
      val origin = row.toSeq.toArray
      for (c <- columns.asScala.toArray) {
        val idx = row.fieldIndex(c)
        if (row.isNullAt(idx)) {
          origin.update(idx, fillVal)
        }
      }
      Row.fromSeq(origin)
    })

    val spark = SparkSession.builder().config(dfUpdated.sparkContext.getConf).getOrCreate()
    spark.createDataFrame(dfUpdated, schema)
  }

  def fillNA(df: DataFrame, fillVal: Int = 0, columns: JList[Int]): Unit = {
    val schema = df.schema
    val dfUpdated = df.rdd.map(row => {
      val origin = row.toSeq.toArray
      for (idx <- columns.asScala.toArray) {
        if (row.isNullAt(idx)) {
          origin.update(idx, fillVal)
        }
      }
      Row.fromSeq(origin)
    })

    val spark = SparkSession.builder().config(dfUpdated.sparkContext.getConf).getOrCreate()
    spark.createDataFrame(dfUpdated, schema)
  }

  private def get_count(rows: Iterator[Row]): Iterator[(Int, Int)] ={
    if(rows.isEmpty){
      Array().iterator
    } else {
      val part_id = TaskContext.get().partitionId()
      Array(Tuple2(part_id, rows.size)).iterator
    }
  }

  def categoryAssignId(df_list: JList[DataFrame], columns: JList[String]): JList[DataFrame] = {
    val idx_df_list = (0 until columns.size).map(x => {
      val col_n = columns.get(x)
      val df = df_list.get(x).withColumn("part_id", spark_partition_id())
      df.cache()
      val count_list: Array[(Int, Int)] = df.rdd.mapPartitions(get_count).collect()
      val base_dict = scala.collection.mutable.Map[Int, Int]()
      var running_sum = 0
      for (count_tuple <- count_list) {
        base_dict += (count_tuple._1 -> running_sum)
        running_sum += count_tuple._2
      }
      val base_dict_bc = df.rdd.sparkContext.broadcast(base_dict)
      val windowSpec  = Window.partitionBy("part_id").orderBy("count")
      val df_row_number = df.withColumn("row_number", row_number.over(windowSpec))
      val get_label = udf((part_id: Int, row_number: Int)=> {
        row_number + base_dict_bc.value.getOrElse(part_id, 0)
      })
      df_row_number.withColumn("id",
        get_label(col("part_id"), col("row_number")))
        .drop("part_id", "row_number", "count")
    })
    idx_df_list.toList.asJava
  }
}

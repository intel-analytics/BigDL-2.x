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
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import java.util.{List => JList}

import org.apache.spark.TaskContext
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, row_number, spark_partition_id, udf}
import org.apache.spark.sql.types.{BooleanType, DataType, LongType, NumericType, StringType}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

object PythonFriesian {
  def ofFloat(): PythonFriesian[Float] = new PythonFriesian[Float]()

  def ofDouble(): PythonFriesian[Double] = new PythonFriesian[Double]()
}

class PythonFriesian[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  val numericTypes: List[String] = List("long", "double", "integer")

  def fillNA(df: DataFrame, fillVal: Any = 0, columns: JList[String] = null): DataFrame = {
    val cols = if(columns == null) {
      df.columns
    } else {
      columns.asScala.toArray
    }

    val cols_idx = cols.map(col_n => {
      val idx = df.columns.indexOf(col_n)
      if(idx == -1) {
        throw new IllegalArgumentException(s"The column name ${col_n} is not exist")
      }
      idx
    })

    fillnaidx(df, fillVal, cols_idx)
  }

  private def fillnaidx(df: DataFrame, fillVal: Any, columns: Array[Int]): DataFrame = {
    val targetType = fillVal match {
      case _: Double | _: Long | _: Int => "numeric"
      case _: String => "string"
      case _: Boolean => "boolean"
      case _ => throw new IllegalArgumentException(
        s"Unsupported value type ${fillVal.getClass.getName} ($fillVal).")
    }

    val schema = df.schema

    val fillValList = columns.map(idx => {
      val matchAndVal = checkTypeAndCast(schema(idx).dataType.typeName, targetType, fillVal)
      if (!matchAndVal._1){
        throw new IllegalArgumentException(s"$targetType is not matched at fillValue")
      }
      matchAndVal._2
    })

    val dfUpdated = df.rdd.map(row => {
      val origin = row.toSeq.toArray
      for ((idx, fillV) <- columns zip fillValList) {
        if (row.isNullAt(idx)) {
          origin.update(idx, fillV)
        }
      }
      Row.fromSeq(origin)
    })

    val spark = SparkSession.builder().config(dfUpdated.sparkContext.getConf).getOrCreate()
    spark.createDataFrame(dfUpdated, schema)
  }

  private def checkTypeAndCast(schemaType: String, targetType: String, fillVal: Any):
  (Boolean, Any) = {
    if (schemaType == targetType) {
      return (true, fillVal)
    } else if (targetType == "numeric"){
      val fillNum = fillVal.asInstanceOf[Number]
      return schemaType match {
          case "long" => (true, fillNum.longValue)
          case "integer" => (true, fillNum.intValue)
          case "double" => (true, fillNum.doubleValue)
          case _ => (false, fillVal)
      }
    }
    (false, fillVal)
  }

  private def getCount(rows: Iterator[Row]): Iterator[(Int, Int)] ={
    if(rows.isEmpty){
      Array[(Int, Int)]().iterator
    } else {
      val part_id = TaskContext.get().partitionId()
      Array(Tuple2(part_id, rows.size)).iterator
    }
  }

  def categoryAssignId(df_list: JList[DataFrame], columns: JList[String]): JList[Object] = {
    val idx_df_list = new java.util.ArrayList[Object]()
    for (x <- 0 until columns.size) {
      val df = df_list.get(x).withColumn("part_id", spark_partition_id())
      df.cache()
      val count_list: Array[(Int, Int)] = df.rdd.mapPartitions(getCount).collect()
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
      idx_df_list.add(df_row_number
        .withColumn("id", get_label(col("part_id"), col("row_number")))
        .drop("part_id", "row_number", "count"))
    }
    idx_df_list
  }
}

package com.intel.analytics.zoo.friesian.feature

import org.apache.spark.TaskContext
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

private[friesian] object Utils {
  def fillNaIndex(df: DataFrame, fillVal: Any, columns: Array[Int]): DataFrame = {
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
      if (!matchAndVal._1) {
        throw new IllegalArgumentException(s"$targetType does not match the type of column " +
          s"${schema(idx).name}")
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

  def checkTypeAndCast(schemaType: String, targetType: String, fillVal: Any):
  (Boolean, Any) = {
    if (schemaType == targetType) {
      return (true, fillVal)
    } else if (targetType == "numeric") {
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

  def getPartitionSize(rows: Iterator[Row]): Iterator[(Int, Int)] = {
    if (rows.isEmpty) {
      Array[(Int, Int)]().iterator
    } else {
      val part_id = TaskContext.get().partitionId()
      Array(Tuple2(part_id, rows.size)).iterator
    }
  }
}

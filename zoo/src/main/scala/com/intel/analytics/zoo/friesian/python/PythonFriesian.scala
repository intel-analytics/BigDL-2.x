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

import java.util

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.friesian.feature.Utils
import java.util.{List => JList}

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{array, col, collect_list, explode, row_number, size, spark_partition_id, struct, udf, log => sqllog}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Column, DataFrame, Row}
import org.apache.spark.ml.linalg.{DenseVector, Vector => MLVector}
import org.apache.spark.ml.feature.MinMaxScaler

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import scala.collection.mutable.WrappedArray
import scala.util.Random

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

    val cols_idx: Array[Int] = Utils.getIndex(df, cols)

    Utils.fillNaIndex(df, fillVal, cols_idx)
  }

  def fillNaInt(df: DataFrame, fillVal: Int = 0, columns: JList[String] = null): DataFrame = {
    val schema = df.schema
    val allColumns = df.columns

    val cols_idx = if (columns == null) {
      schema.zipWithIndex.filter(pair => pair._1.dataType.typeName == "integer")
        .map(pair => pair._2)
    } else {
      val cols = columns.asScala.toList
      cols.map(col_n => {
        val idx = allColumns.indexOf(col_n)
        if (idx == -1) {
          throw new IllegalArgumentException(s"The column name ${col_n} does not exist")
        }
        if (schema(idx).dataType.typeName != "integer") {
          throw new IllegalArgumentException(s"Only columns of IntegerType are supported, but " +
            s"the type of column ${col_n} is ${schema(idx).dataType.typeName}")
        }
        idx
      })
    }

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

  def generateStringIdx(df: DataFrame, columns: JList[String], frequencyLimit: String = null)
  : JList[DataFrame] = {
    var default_limit: Option[Int] = None
    val freq_map = scala.collection.mutable.Map[String, Int]()
    if (frequencyLimit != null) {
      val freq_list = frequencyLimit.split(",")
      for (fl <- freq_list) {
        val frequency_pair = fl.split(":")
        if (frequency_pair.length == 1) {
          default_limit = Some(frequency_pair(0).toInt)
        } else if (frequency_pair.length == 2) {
          freq_map += (frequency_pair(0) -> frequency_pair(1).toInt)
        }
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

      val windowSpec = Window.partitionBy("part_id").orderBy("count")
      val df_with_part_id = df_col_filtered.withColumn("part_id", spark_partition_id())
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

  def log(df: DataFrame, columns: JList[String], clipping: Boolean = true): DataFrame = {
    val colsIdx = Utils.getIndex(df, columns.asScala.toArray)
    for(i <- 0 until columns.size()) {
      val colName = columns.get(i)
      val colType = df.schema(colsIdx(i)).dataType.typeName
      if (!Utils.checkColumnNumeric(df, colName)) {
        throw new IllegalArgumentException(s"Unsupported data type $colType of column $colName")
      }
    }

    var resultDF = df
    val zeroThreshold = (value: Int) => {
      if (value < 0) 0 else value
    }

    val zeroThresholdUDF = udf(zeroThreshold)
    for (i <- 0 until columns.size()) {
      val colName = columns.get(i)
      if (clipping) {
        resultDF = resultDF.withColumn(colName, sqllog(zeroThresholdUDF(col(colName)) + 1))
      } else {
        resultDF = resultDF.withColumn(colName, sqllog(col(colName)))
      }
    }
    resultDF
  }

  def clip(df: DataFrame, columns: JList[String], min: Any = null, max: Any = null):
  DataFrame = {
    if (min == null && max == null) {
      throw new IllegalArgumentException(s"min and max cannot be both null")
    }
    var resultDF = df
    val cols = columns.asScala.toArray
    val colsType = Utils.getIndex(df, cols).map(idx => df.schema(idx).dataType.typeName)
    (cols zip colsType).foreach(nameAndType => {
      if (!Utils.checkColumnNumeric(df, nameAndType._1)) {
        throw new IllegalArgumentException(s"Unsupported data type ${nameAndType._2} of " +
          s"column ${nameAndType._1}")
      }
    })

    for(i <- 0 until columns.size()) {
      val colName = columns.get(i)
      val colType = colsType(i)

      val minVal = Utils.castNumeric(min, colType)
      val maxVal = Utils.castNumeric(max, colType)

      val clipFuncUDF = colType match {
        case "long" => udf(Utils.getClipFunc[Long](minVal, maxVal, colType))
        case "integer" => udf(Utils.getClipFunc[Int](minVal, maxVal, colType))
        case "double" => udf(Utils.getClipFunc[Double](minVal, maxVal, colType))
        case _ => throw new IllegalArgumentException(s"Unsupported data type $colType of column" +
          s" $colName")
      }
      resultDF = resultDF.withColumn(colName, clipFuncUDF(col(colName)))
    }
    resultDF
  }

  def crossColumns(df: DataFrame,
                   crossCols: JList[JList[String]],
                   bucketSizes: JList[Int]): DataFrame = {
    def crossColumns(bucketSize: Int) = udf((cols: collection.mutable.WrappedArray[Any]) => {
      Utils.hashBucket(cols.mkString("_"), bucketSize = bucketSize)
    })

    var resultDF = df
    for (i <- 0 until crossCols.size()) {
      resultDF = resultDF.withColumn(crossCols.get(i).asScala.toList.mkString("_"),
        crossColumns(bucketSizes.get(i))(
          array(crossCols.get(i).asScala.toArray.map(x => col(x)): _*)
        ))
    }
    resultDF
  }

  def addHistSeq(df: DataFrame,
                 cols: JList[String],
                 userCol: String,
                 timeCol: String,
                 minLength: Int,
                 maxLength: Int): DataFrame = {

    df.sparkSession.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
    val colNames: Array[String] = cols.asScala.toArray
    val colsWithType = df.schema.fields.filter(x => colNames.contains(x.name))
    val schema = ArrayType(StructType(colsWithType.flatMap(c =>
      Seq(c, StructField(c.name + "_hist_seq", ArrayType(c.dataType))))))

    val genHisUDF = udf(f = (his_collect: Seq[Row]) => {
      val full_rows: Array[Row] = his_collect.sortBy(x => x.getAs[Long](timeCol)).toArray
      val n = full_rows.length

      val result: Seq[Row] = (minLength to n - 1).map(i => {
        val lowerBound = if (i < maxLength) {
          0
        } else {
          i - maxLength
        }

        val rowValue = colsWithType.flatMap(col => {
          col.dataType.typeName match {
            case "integer" => get1row[Int](full_rows, col.name, i, lowerBound)
            case "double" => get1row[Double](full_rows, col.name, i, lowerBound)
            case "float" => get1row[Float](full_rows, col.name, i, lowerBound)
            case "long" => get1row[Long](full_rows, col.name, i, lowerBound)
            case _ => throw new IllegalArgumentException(
              s"Unsupported data type ${col.dataType.typeName} in addHistSeq")
          }
        })
        Row.fromSeq(rowValue)
      })
      result
    }, schema)

    val collectColumns = colNames.map(c => col(c)) ++ Seq(col(timeCol))

    df.groupBy(userCol)
      .agg(collect_list(struct(collectColumns: _*)).as("friesian_his_collect"))
      .withColumn("friesian_history", explode(genHisUDF(col("friesian_his_collect"))))
      .select(userCol, "friesian_history.*")
  }

  private def get1row[T](full_rows: Array[Row], colName: String, index: Int, lowerBound: Int) = {
    val colValue = full_rows(index).getAs[T](colName)
    val historySeq = full_rows.slice(lowerBound, index).map(row => row.getAs[T](colName))
    Seq(colValue, historySeq)
  }

  def mask(df: DataFrame, cols: JList[String], maxLength: Int): DataFrame = {
    val colDataTypes = df.schema.fields.filter(x => cols.contains(x.name))
      .map(x => x.dataType).distinct
    require(colDataTypes.length == 1, "dataTypes should be the same at each operation")

    colDataTypes(0) match {
      case ArrayType(IntegerType, _) =>
        df.sqlContext.udf.register("mask", Utils.maskArr[Int])
      case ArrayType(LongType, _) =>
        df.sqlContext.udf.register("mask", Utils.maskArr[Long])
      case ArrayType(DoubleType, _) =>
        df.sqlContext.udf.register("mask", Utils.maskArr[Double])
      case _ => throw new IllegalArgumentException(s"Unsupported data type $colDataTypes in mask")
    }

    df.createOrReplaceTempView("tmp")

    val selectStatement = cols.toArray().map(c =>
      s"mask($maxLength, $c) as $c" + "_mask").mkString(",")

    df.sqlContext.sql(s"select *, $selectStatement from tmp")
  }

  def addNegHisSeq(df: DataFrame, itemSize: Int,
                   historyCol: String,
                   negNum: Int = 5): DataFrame = {

    df.sparkSession.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
    val itemType = df.select(explode(col(historyCol))).schema.fields(0).dataType
    val schema = ArrayType(ArrayType(itemType))

    val negativeUdf = udf(Utils.addNegativeList(negNum, itemSize), schema)

    df.withColumn("neg_" + historyCol, negativeUdf(col(historyCol)))
  }

  def addNegSamples(df: DataFrame,
                    itemSize: Int,
                    itemCol: String = "item",
                    labelCol: String = "label",
                    negNum: Int = 1): DataFrame = {

    df.sparkSession.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
    val itemType = df.select(itemCol).schema.fields(0).dataType
    val schema = ArrayType(StructType(Seq(StructField(itemCol, itemType),
      StructField(labelCol, itemType))))

    val negativeUdf = udf(Utils.addNegtiveItem(negNum, itemSize), schema)

    val negativedf = df.withColumn("item_label", explode(negativeUdf(col(itemCol))))

    val selectColumns = df.columns.filter(x => x != itemCol)
      .map(ele => col(ele)) ++ Seq(col("item_label.*"))

    negativedf.select(selectColumns: _*)
  }

  def postPad(df: DataFrame, cols: JList[String], maxLength: Int = 100): DataFrame = {
    val colDataTypes = df.schema.fields.filter(x =>
      cols.contains(x.name)).map(x => x.dataType).distinct
    colDataTypes.map(dataType =>
      dataType match {
        case ArrayType(IntegerType, _) =>
          df.sqlContext.udf.register("pad_array", Utils.padArr[Int])
        case ArrayType(LongType, _) =>
          df.sqlContext.udf.register("pad_array", Utils.padArr[Long])
        case ArrayType(DoubleType, _) =>
          df.sqlContext.udf.register("pad_array", Utils.padArr[Double])
        case ArrayType(ArrayType(IntegerType, _), _) =>
          df.sqlContext.udf.register("pad_matrix", Utils.padMatrix[Int])
        case ArrayType(ArrayType(LongType, _), _) =>
          df.sqlContext.udf.register("pad_matrix", Utils.padMatrix[Long])
        case ArrayType(ArrayType(DoubleType, _), _) =>
          df.sqlContext.udf.register("pad_matrix", Utils.padMatrix[Double])
        case _ => throw new IllegalArgumentException(s"Unsupported data type $dataType in postPad")
      })

    df.createOrReplaceTempView("tmp")

    val selectStatement = df.schema.fields
      .filter(x => cols.contains(x.name)).map(x => {
      val c = x.name
      if (x.dataType.toString.contains("ArrayType(ArrayType")) {
        s"pad_matrix($maxLength, $c) as $c"
      } else {
        s"pad_array($maxLength, $c) as $c"
      }
    }).mkString(",")

    val leftCols = df.columns.filter(x => !cols.contains(x)).mkString(",")

    df.sqlContext.sql(s"select $leftCols, $selectStatement from tmp")
  }

  def fillMedian(df: DataFrame, columns: JList[String] = null): DataFrame = {
    val cols = if (columns == null) {
      df.columns.filter(column => Utils.checkColumnNumeric(df, column))
    } else {
      columns.asScala.toArray
    }

    val colsIdx = Utils.getIndex(df, cols)
    val medians = Utils.getMedian(df, cols)
    val idxMedians = (colsIdx zip medians).map(idxMedian => {
      if (idxMedian._2 == null) {
        throw new IllegalArgumentException(
          s"Cannot compute the median of column ${cols(idxMedian._1)} " +
            s"since it contains only null values.")
      }
      val colType = df.schema(idxMedian._1).dataType.typeName
      colType match {
        case "long" => (idxMedian._1, idxMedian._2.asInstanceOf[Double].longValue)
        case "integer" => (idxMedian._1, idxMedian._2.asInstanceOf[Double].intValue)
        case "double" => (idxMedian._1, idxMedian._2.asInstanceOf[Double])
        case _ => throw new IllegalArgumentException(
          s"Unsupported value type $colType of column ${cols(idxMedian._1)}.")
      }
    })

    val dfUpdated = df.rdd.map(row => {
      val origin = row.toSeq.toArray
      for ((idx, fillV) <- idxMedians) {
        if (row.isNullAt(idx)) {
          origin.update(idx, fillV)
        }
      }
      Row.fromSeq(origin)
    })

    val spark = df.sparkSession
    spark.createDataFrame(dfUpdated, df.schema)
  }

  /* ---- Stat Operator ---- */

  def median(df: DataFrame, columns: JList[String] = null, relativeError: Double = 0.00001):
  DataFrame = {
    val cols = if (columns == null) {
      df.columns.filter(column => Utils.checkColumnNumeric(df, column))
    } else {
      columns.asScala.toArray
    }

    Utils.getIndex(df, cols)  // checks if `columns` exist in `df`
    val medians = Utils.getMedian(df, cols, relativeError)
    val medians_data = (cols zip medians).map(cm => Row.fromSeq(Array(cm._1, cm._2)))
    val spark = df.sparkSession
    val schema = StructType(Array(
      StructField("column", StringType, nullable = true),
      StructField("median", DoubleType, nullable = true)
    ))
    spark.createDataFrame(spark.sparkContext.parallelize(medians_data), schema)
  }

  def normalizeArray(df: DataFrame, column: String): DataFrame = {
    val toVector = udf((arr: Seq[Any]) => {
      val doubleArray = arr.map {
        case f: Float => f.toDouble
        case i: Int => i.toDouble
        case l: Long => l.toDouble
        case d: Double => d
      }
      new DenseVector(doubleArray.toArray)
    })

    val vectoredDF = df.withColumn(column, toVector(col(column)))
    val scaler = new MinMaxScaler()
        .setInputCol(column)
        .setOutputCol("scaled");
    val toArray = udf((vec: MLVector) => vec.toArray.map(_.toFloat))
    val resultDF = scaler.fit(vectoredDF).transform(vectoredDF)
      .withColumn(column, toArray(col("scaled"))).drop("scaled")
    resultDF
  }
}

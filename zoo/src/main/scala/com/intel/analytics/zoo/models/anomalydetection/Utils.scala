package com.intel.analytics.zoo.models.anomalydetection

import com.intel.analytics.bigdl.dataset.Sample
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{avg, col, udf}

object Utils {

  def standardScaleHelper(df: DataFrame, colName: String) = {

    val mean = df.select(colName).agg(avg(col(colName))).collect()(0).getDouble(0)

    val stddevUdf = udf((num: Float) => (num - mean) * (num - mean))

    val stddev = Math.sqrt(df.withColumn("stddev", stddevUdf(col(colName)))
      .agg(avg(col("stddev"))).collect()(0).getDouble(0))

    val scaledUdf = udf((num: Float) => ((num - mean) / stddev).toFloat)

    df.withColumn(colName, scaledUdf(col(colName)))
  }

  def standardScale(df: DataFrame, fields: Seq[String], index: Int = 0): DataFrame = {

    if (index == fields.length) {
      df
    } else {
      val colDf = standardScaleHelper(df, fields(index))
      standardScale(colDf, fields, index + 1)
    }
  }

  def trainTestSplit(unrolled: RDD[FeatureLabelIndex[Float]], testSize: Int = 1000) = {

    val cutPoint = unrolled.count() - testSize

    val train = AnomalyDetector.toSampleRdd(unrolled.filter(x => x.index < cutPoint))
    val test = AnomalyDetector.toSampleRdd(unrolled.filter(x => x.index >= cutPoint))

    (train, test)
  }

  def trainTestSplit(unrolled: RDD[FeatureLabelIndex[Float]], testSize: Float)
  : (RDD[Sample[Float]], RDD[Sample[Float]]) = {

    val totalSize = unrolled.count()
    val testSizeInt = (totalSize * testSize).toInt

    trainTestSplit(unrolled, testSizeInt)
  }


}

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

package com.intel.analytics.zoo.examples.anomalydetection

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.anomalydetection.AnomalyDetector
import com.intel.analytics.zoo.models.anomalydetection.AnomalyDetector._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import scopt.OptionParser
import org.joda.time.format.DateTimeFormat

case class Taxi(ts: String, value: Float)

case class LocalParams(val inputDir: String = "./data/NAB/nyc_taxi/",
                       val batchSize: Int = 1024,
                       val nEpochs: Int = 20,
                       val learningRate: Double = 1e-3,
                       val learningRateDecay: Double = 1e-6
                      )

object AnomalyDetectorExample {

  def main(args: Array[String]): Unit = {

    val defaultParams = LocalParams()

    val parser = new OptionParser[LocalParams]("NCF Example") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x.toInt))
      opt[Int]('e', "nEpochs")
        .text("epoch numbers")
        .action((x, c) => c.copy(nEpochs = x))
      opt[Double]('l', "lRate")
        .text("learning rate")
        .action((x, c) => c.copy(learningRate = x.toDouble))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: LocalParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("AnomalyDetectionExample").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val featureDF = loadData(sqlContext, param.inputDir)
    val unrollLength = 50
    val (trainRdd, testRdd) = assemblyFeature(featureDF.select("value", "hour", "awake"), scale = true, unrollLength, testdataSize = 1000)

    val model = AnomalyDetector[Float](inputShape = Shape(50, 3)).buildModel().asInstanceOf[Sequential[Float]]

    model.compile(loss = "mse", optimizer = "rmsprop")

    val batchSize = Math.min(param.batchSize, trainRdd.count().toInt / 8 * 8)

    model.fit(trainRdd, batchSize = batchSize, nbEpoch = param.nEpochs)
    val predictions = model.predict(testRdd, batchSize = batchSize)

    val yPredict: RDD[Float] = predictions.map(x => x.toTensor.toArray()(0))
    val yTruth: RDD[Float] = testRdd.map(x => x.label.toArray()(0))
    val anomolies = detectAnomalies[Float](yPredict, yTruth, 5)
    anomolies.map(x => x._1 + "," + x._2 + "," + x._3).coalesce(1).saveAsTextFile("/Users/guoqiong/intelWork/projects/BaoSight/result/nyc/test")
  }


  def loadData(sqlContext: SQLContext, dataPath: String) = {

    @transient lazy val formatter = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss")

    import sqlContext.implicits._
    val df = sqlContext.read.text(dataPath + "/nyc_taxi.csv").as[String]
      .rdd.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter
    }
      .map(x => {
        val line = x.split(",")
        Taxi(line(0), line(1).toFloat)
      }).toDF()
    // hour, awake (hour>=6 <=23)
    val hourUdf = udf((time: String) => {
      val dt = formatter.parseDateTime(time)
      dt.hourOfDay().get()
    })

    val awakeUdf = udf((hour: Int) => if (hour >= 6 && hour <= 23) 1 else 0)

    val featureDF = df.withColumn("hour", hourUdf(col("ts")))
      .withColumn("awake", awakeUdf(col("hour")))
      .drop("ts")
      .select("value", "hour", "awake")

    featureDF.show(5, false)
    println("tatal count: " + featureDF.count())

    featureDF
  }

  def assemblyFeature(df: DataFrame, scale: Boolean = true, unrollLength: Int, testdataSize: Int = 1000): (RDD[Sample[Float]], RDD[Sample[Float]]) = {

//    val scaledDF = if (scale) standardScale(df, Seq("value", "hour", "awake")) else df
//    val dataRdd: RDD[Array[Float]] = scaledDF.rdd.map(row => Array
//    (row.getAs[Float](0), row.getAs[Float](1), row.getAs[Float](2)))

    val vectorizeUdf = udf((value:Double,hour:Double,awake:Double) => Vectors.dense(Array(value,hour,awake)))
    val featureDF = df.withColumn("features",vectorizeUdf(col("value"), col("hour"),col("awake")))

    val scaler = new org.apache.spark.ml.feature.StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(df)

    // Normalize each feature to have unit standard deviation.
    val scaledDF = scalerModel.transform(df)

    val dataRdd: RDD[Array[Float]] = scaledDF.select("scaledFeatures").
      rdd.map(row => row.getAs[org.apache.spark.mllib.linalg.Vector](0).toArray.map(x=> x.toFloat))

    val unrollData = distributeUnrollAll(dataRdd, unrollLength)

    val cutPoint = unrollData.count() - testdataSize

    val train: RDD[Sample[Float]] = toSampleRdd(unrollData.filter(x => x.index < cutPoint))
    val test = toSampleRdd(unrollData.filter(x => x.index >= cutPoint))

    (train, test)
  }

  def assemblyFeature(df: DataFrame, scale: Boolean, unrollLength: Int, testdataSize: Float): (RDD[Sample[Float]], RDD[Sample[Float]]) = {
    val totalCount = df.count()
    val testSizeInt = (totalCount * testdataSize).toInt

    val (train, test) = assemblyFeature(df, scale, unrollLength: Int, testSizeInt)
    (train, test)
  }
}

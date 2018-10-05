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
import com.intel.analytics.zoo.models.anomalydetection.{AnomalyDetector, FeatureLabelIndex}
import com.intel.analytics.zoo.models.anomalydetection.AnomalyDetector._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import scopt.OptionParser
import org.joda.time.format.DateTimeFormat

case class Taxi(ts: String, value: Float)

case class LocalParams(val inputDir: String = "./data/NAB/nyc_taxi/",
                       val batchSize: Int = 1024,
                       val nEpochs: Int = 20
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
    conf.setAppName("AnomalyDetectionExample")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val featureDF = loadData(sqlContext, param.inputDir)
    val featureShape = Shape(50, 3)
    val (trainRdd, testRdd) = assemblyFeature(featureDF, ifScale = true, 50, testdataSize = 1000)

    val model = AnomalyDetector[Double](featureShape).buildModel().asInstanceOf[Sequential[Float]]
    model.compile(loss = "mse", optimizer = "rmsprop")
    model.fit(trainRdd, batchSize = param.batchSize, nbEpoch = param.nEpochs)
    val predictions = model.predict(testRdd)

    val yPredict: RDD[Float] = predictions.map(x => x.toTensor.toArray()(0))
    val yTruth: RDD[Float] = testRdd.map(x => x.label.toArray()(0))
    val anomolies = detectAnomalies(yPredict, yTruth, 5)
    anomolies.take(5).foreach(println)
  }

  def loadData(sqlContext: SQLContext, dataPath: String) = {

    @transient lazy val formatter = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss")
    import sqlContext.implicits._

    val df = sqlContext.sparkContext.textFile(dataPath + "/nyc_taxi.csv")
      .mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter.drop(1) else iter)
      .map(x => {
        val line = x.split(",")
        Taxi(line(0), line(1).toFloat)
      }).toDF()

    val hourUDF = udf((time: String) => (formatter.parseDateTime(time).hourOfDay().get()))
    val awakeUDF = udf((hour: Int) => if (hour >= 6 && hour <= 23) 1 else 0)
    val vectorUDF = udf((col1: Double, col2: Double, col3: Double) =>
      Vectors.dense(Array(col1, col2, col3)))

    val featureDF = df.withColumn("hour", hourUDF(col("ts")))
      .withColumn("awake", awakeUDF(col("hour")))
      .select("value", "hour", "awake")

    featureDF.withColumn("features", vectorUDF(col("value"), col("hour"), col("awake")))
  }

  def assemblyFeature(featureDF: DataFrame,
                      ifScale: Boolean = true,
                      unrollLength: Int,
                      testdataSize: Int = 1000
                     ): (RDD[Sample[Float]], RDD[Sample[Float]]) = {

    val scaler = new org.apache.spark.ml.feature.StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)

    val scaledDF = if (ifScale) {
      val scalerModel = scaler.fit(featureDF)
      scalerModel.transform(featureDF).select("scaledFeatures")
    } else {
      featureDF.select("feature")
    }

    val dataRdd: RDD[Array[Float]] = scaledDF.rdd
      .map(row => row.getAs[org.apache.spark.ml.linalg.DenseVector](0).toArray.map(x => x.toFloat))

    val unrolled: RDD[FeatureLabelIndex[Float]] = unroll(dataRdd, unrollLength)

    val cutPoint = unrolled.count() - testdataSize

    val train: RDD[Sample[Float]] = toSampleRdd(unrolled.filter(x => x.index < cutPoint))
    val test = toSampleRdd(unrolled.filter(x => x.index >= cutPoint))

    (train, test)
  }

  def assemblyFeature(featureDF: DataFrame,
                      ifScale: Boolean,
                      unrollLength: Int,
                      testdataSize: Float
                     ): (RDD[Sample[Float]], RDD[Sample[Float]]) = {
    val totalCount = featureDF.count()
    val testSizeInt = (totalCount * testdataSize).toInt

    val (train, test) = assemblyFeature(featureDF, ifScale, unrollLength: Int, testSizeInt)
    (train, test)
  }
}

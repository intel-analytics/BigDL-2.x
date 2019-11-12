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

package com.intel.analytics.zoo.examples.recommendation

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.models.recommendation.SessionRecommender
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import scopt.OptionParser
import com.intel.analytics.zoo.models.recommendation.Utils._

import scala.collection.mutable

case class SessionParams(sessionLength: Int = 5,
                         historyLength: Int = 5,
                         maxEpoch: Int = 10,
                         batchSize: Int = 1024,
                         embedOutDim: Int = 100,
                         learningRate: Double = 3e-3,
                         learningRateDecay: Double = 5e-5,
                         input: String = " ",
                         outputDir: String = " ")

object SessionRecExp {

  def main(args: Array[String]): Unit = {

    val defaultParams = SessionParams()

    val parser = new OptionParser[SessionParams]("SessionRecExample") {
      opt[String]("input")
        .text(s"input")
        .required()
        .action((x, c) => c.copy(input = x))
      opt[String]("outputDir")
        .text(s"outputDir")
        .required()
        .action((x, c) => c.copy(outputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]('e', "maxEpoch")
        .text(s"max epoch, default is 10")
        .action((x, c) => c.copy(maxEpoch = x))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: SessionParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("SessionRecExp")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (sessionDF, itemCount, itemStart) = loadPublicData(sqlContext, params)
    println(itemCount)
    println(itemStart)

    val trainSample =
      assemblyFeature(sessionDF, params.sessionLength, includeHistory = true, params.historyLength)
    val Array(trainRdd, testRdd) = trainSample.randomSplit(Array(0.8, 0.2), 100)
    testRdd.cache()

    val model = SessionRecommender[Float](
      itemCount = itemCount,
      itemEmbed = params.embedOutDim,
      sessionLength = params.sessionLength,
      includeHistory = true,
      mlpHiddenLayers = Array(20, 10),
      historyLength = params.historyLength)

    val optimMethod = new RMSprop[Float](
      learningRate = params.learningRate,
      learningRateDecay = params.learningRateDecay)

    model.compile(
      optimizer = optimMethod,
      loss = new SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      metrics = List(new Top5Accuracy[Float]()))

    model.fit(trainRdd, batchSize = params.batchSize, validationData = testRdd)

    model.saveModule(params.outputDir, null, overWrite = true)
    println("Model has been saved")

    val loaded = SessionRecommender.loadModel(params.outputDir)
    val results = loaded.predict(testRdd)
    results.take(5).foreach(println)

    val resultsClass = loaded.predictClasses(testRdd, zeroBasedLabel = false)
    resultsClass.take(5).foreach(println)

    val recommendations = model.recommendForSession(testRdd, 5, false)

    recommendations.take(20).map( x => println(x.toList))

  }

  def loadPublicData(sqlContext: SQLContext, params: SessionParams): (DataFrame, Int, Int) = {

    val toFloat = {
      val func: ((mutable.WrappedArray[Double]) => mutable.WrappedArray[Float]) =
        (seq => seq.map(_.toFloat))
      udf(func)
    }

    val sessionDF = sqlContext.read
      .options(Map("header" -> "false", "delimiter" -> ",")).json(params.input)
      .withColumn("session", toFloat(col("ATC_SEQ")))
      .withColumn("purchase_history", toFloat(col("PURCH_HIST")))
      .select("session", "purchase_history")

    val atcMax = sessionDF.rdd
      .map(_.getAs[mutable.WrappedArray[Float]]("session").max).collect().max.toInt
    val purMax = sessionDF.rdd
      .map(_.getAs[mutable.WrappedArray[Float]]("purchase_history").max)
      .collect().max.toInt
    val itemCount = Math.max(atcMax, purMax)

    val atcMin = sessionDF.rdd
      .map(_.getAs[mutable.WrappedArray[Float]]("session").min).collect().min.toInt
    val purMin = sessionDF.rdd
      .map(_.getAs[mutable.WrappedArray[Float]]("purchase_history").min).collect().min.toInt
    val itemStart = Math.max(atcMin, purMin)

    (sessionDF, itemCount, itemStart)
  }

  def assemblyFeature(sessionDF: DataFrame,
                      sessionLength: Int,
                      includeHistory: Boolean = true,
                      historyLength: Int = 10): RDD[Sample[Float]] = {

    val expandedDF = slideSession(sessionDF, sessionLength)

    val padSession = udf(prePadding(sessionLength))

    val sessionPaddedDF = expandedDF
      .withColumn("sessionPadded", padSession(col("session")))
      .drop("session")
      .withColumnRenamed("sessionPadded", "session")

    val paddedDF = if (includeHistory) {
      val padHistory = udf(prePadding(historyLength))
      sessionPaddedDF
        .withColumn("purchasePadded", padHistory(col("purchase_history")))
        .drop("purchase_history")
        .withColumnRenamed("purchasePadded", "purchase_history")
    }
    else sessionPaddedDF

    // dataFrame to rdd of sample
    val samples = paddedDF.rdd.map(r => {
      row2sampleSession(r, sessionLength, true, historyLength)
    })
    samples
  }
}

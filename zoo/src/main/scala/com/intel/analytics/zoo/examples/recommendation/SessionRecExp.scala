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

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.{Sample, TensorSample}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
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
                         batchSize: Int = 1280,
                         embedOutDim: Int = 20,
                         learningRate: Double = 1e-3,
                         learningRateDecay: Double = 1e-6,
                         inputDir: String = "/Users/guoqiong/intelWork/projects/officeDepot/sampleData/",
                         outputDir: String = "./model/",
                         fileName: String = "atcHistory.json",
                         modelName: String = "sessionRecommender"
                        )

object SessionRecExp {

  val currentDir: String = Paths.get(".").toAbsolutePath + "/"

  def main(args: Array[String]): Unit = {

    val defaultParams = SessionParams()

    val parser = new OptionParser[SessionParams]("SessionRecExample") {
      opt[String]("inputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(inputDir = x))
      opt[String]("outputDir")
        .text(s"inputDir")
        .action((x, c) => c.copy(outputDir = x))
      opt[Int]('b', "batchSize")
        .text(s"batchSize")
        .action((x, c) => c.copy(batchSize = x))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(params: SessionParams): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)

    // construct BigDL session
    val conf = new SparkConf()
      .setAppName("SessionRecExp")
      .setMaster("local[*]")

    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (sessionDF, itemCount) = loadPublicData(sqlContext, params)
    val trainSample = assemblyFeature(sessionDF, params.sessionLength, includeHistory = true, params.historyLength)
    val Array(trainRdd, testRdd) = trainSample.randomSplit(Array(0.8, 0.2), 100)

    val model = SessionRecommender[Float](
      itemCount = itemCount,
      itemEmbed = params.embedOutDim,
      mlpHiddenLayers = Array(20, 10),
      seqLength = params.sessionLength,
      includeHistory = true,
      hisLength = params.historyLength)

    val optimMethod = new RMSprop[Float](
      learningRate = params.learningRate,
      learningRateDecay = params.learningRateDecay)

    model.compile(
      optimizer = optimMethod,
      loss = new SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      metrics = List(new Top5Accuracy[Float]()))

    model.fit(trainRdd, batchSize = params.batchSize, validationData = testRdd)


    model.saveModule(params.inputDir + params.modelName, null, overWrite = true)
    println("Model has been saved")

  }

  def loadPublicData(sqlContext: SQLContext, params: SessionParams): (DataFrame, Int) = {

    val toFloat = {
      val func: ((mutable.WrappedArray[Double]) => mutable.WrappedArray[Float]) = (seq => seq.map(_.toFloat))
      udf(func)
    }

    val sessionDF = sqlContext.read
      .options(Map("header" -> "false", "delimiter" -> ",")).json(params.inputDir + params.fileName)
      .withColumn("session", toFloat(col("ATC_SEQ")))
      .withColumn("purchase_history", toFloat(col("PURCH_HIST")))
      .select("session", "purchase_history")

    val atcMax = sessionDF.rdd.map(_.getAs[mutable.WrappedArray[Float]]("session").max).collect().max.toInt
    val purMax = sessionDF.rdd.map(_.getAs[mutable.WrappedArray[Float]]("purchase_history").max).collect().max.toInt
    val itemCount = Math.max(atcMax, purMax)
    (sessionDF, itemCount)
  }

  def assemblyFeature(sessionDF: DataFrame,
                      sessionLength: Int,
                      includeHistory: Boolean = true,
                      historyLength: Int = 10): RDD[Sample[Float]] = {

    val expandedDF = slideSession(sessionDF, sessionLength)

    expandedDF.show(10, false)
    val padSession = udf(prePadding(sessionLength))

    val sessionPaddedDF = expandedDF
      .withColumn("sessionPadded", padSession(col("session")))
      .drop("session")
      .withColumnRenamed("sessionPadded", "session")

    sessionPaddedDF.show(10, false)
    val paddedDF = if (includeHistory) {
      val padHistory = udf(prePadding(historyLength))
      sessionPaddedDF
        .withColumn("purchasePadded", padHistory(col("purchase_history")))
        .drop("purchase_history")
        .withColumnRenamed("purchasePadded", "purchase_history")
    }
    else sessionPaddedDF

    println("----------------------------")
    paddedDF.show(10, false)

    // dataFrame to rdd of sample
    val trainSample = paddedDF.rdd.map(r => {
      rows2sample(r, sessionLength, true, historyLength)
    })

    println(trainSample.take(10))
    println("Sample feature print: \n" + trainSample.take(1).head.feature(0))
    println("Sample feature print: \n" + trainSample.take(1).head.feature(1))
    println("Sample label print: \n" + trainSample.take(1).head.label())

    trainSample
  }

}

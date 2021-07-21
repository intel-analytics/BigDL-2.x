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
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, Top1Accuracy}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation.{NeuralCF, UserItemFeature, Utils}
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.functions._
import scopt.OptionParser

case class NeuralCFParams(val inputDir: String = "./data/ml-1m",
                          val batchSize: Int = 2800,
                          val nEpochs: Int = 10,
                          val learningRate: Double = 1e-3,
                          val learningRateDecay: Double = 1e-6,
                          val numClasses: Int = 5,
                          val fileName: String = "/ratings.dat",
                          val delimiter: String = "::"
                         )
case class Rating(userId: Int, itemId: Int, label: Int)
case class RatingFloat(userId: Int, itemId: Int, label: Float)

object NeuralCFexample {

  def main(args: Array[String]): Unit = {

    val defaultParams = NeuralCFParams()

    val parser = new OptionParser[NeuralCFParams]("NCF Example") {
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
      opt[Int]('c', "numClasses")
        .text("range of label")
        .action((x, c) => c.copy(numClasses = x))
      opt[String]("fileName")
        .text(s"fileName")
        .action((x, c) => c.copy(fileName = x))
      opt[String]("delimiter")
        .text(s"delimiter")
        .action((x, c) => c.copy(delimiter = x))
    }

    parser.parse(args, defaultParams).map {
      params =>
        run(params)
    } getOrElse {
      System.exit(1)
    }
  }

  def run(param: NeuralCFParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf()
    conf.setAppName("NCFExample").set("spark.sql.crossJoin.enabled", "true")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    println(param)

    val (ratings, userCount, itemCount) = loadPublicData(sqlContext, param.inputDir, param.fileName, param.delimiter)
    println("Loadpublicdata END")
    ratings.show()
    val isImplicit = false
    val ncf = NeuralCF[Float](
      userCount = userCount,
      itemCount = itemCount,
      numClasses = param.numClasses,
      userEmbed = 20,
      itemEmbed = 20,
      hiddenLayers = Array(20, 10))

    val pairFeatureRdds: RDD[UserItemFeature[Float]] =
      assemblyFeature(isImplicit, ratings, userCount, itemCount)
    println("after assembly")
    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      pairFeatureRdds.randomSplit(Array(0.8, 0.2))
    println("after randomsplit")
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    println("after training")
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)
    println("after validation")
    val optimMethod = new Adam[Float](
      learningRate = param.learningRate,
      learningRateDecay = param.learningRateDecay)

    ncf.compile(optimizer = optimMethod,
      loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      metrics = List(new Top1Accuracy[Float]()))
    println("after compiling")
    ncf.fit(trainRdds, batchSize = param.batchSize,
      nbEpoch = param.nEpochs, validationData = validationRdds)
    println("after training")
    val results = ncf.predict(validationRdds)
    results.take(5).foreach(println)
    val resultsClass = ncf.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)
    println("after validation")
    val userItemPairPrediction = ncf.predictUserItemPair(validationpairFeatureRdds)

    userItemPairPrediction.take(5).foreach(println)

    val userRecs = ncf.recommendForUser(validationpairFeatureRdds, 3)
    val itemRecs = ncf.recommendForItem(validationpairFeatureRdds, 3)

    userRecs.take(10).foreach(println)
    itemRecs.take(10).foreach(println)
    ncf.summary()

    println("finished...")
    sc.stop()
  }

  def readTextAndParse( sqlContext: SQLContext,
                        dataPath: String,
                        dataFileName: String,
                        delimiter: String,
                        userIdx:  Int,
                        itemIdx: Int,
                        labelIdx: Int
                      ): (DataFrame, Int, Int) = {
    import sqlContext.implicits._
    println(dataFileName)
    dataFileName match {
      case "/netflix_short.dat" =>
        val ratings = sqlContext.read.format("csv").option("delimiter", ",").load(dataPath + dataFileName).as[String]
          .map(x => {
            val line = x.split(delimiter).map(n => n.toInt)
            Rating(line(userIdx), line(itemIdx), line(labelIdx))
          }).toDF()
        val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
        val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))
        (ratings, userCount, itemCount)

      case "/ratings.dat" =>
      val ratings = sqlContext.read.text(dataPath + dataFileName).as[String]
        .map(x => {
          val line = x.split(delimiter).map(n => n.toInt)
          Rating(line(userIdx), line(itemIdx), line(labelIdx))
        }).toDF()
      val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
      val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))
      (ratings, userCount, itemCount)

      case "/amazon_short.dat" =>
      val ratings = sqlContext.read.text(dataPath + dataFileName).as[String]
        .map(x => {
          val line = x.split(delimiter).map(n => n.toInt)
          Rating(line(userIdx), line(itemIdx), line(labelIdx))
        }).toDF()
      val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
      val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))
      (ratings, userCount, itemCount)

      case "/book_short_orig.dat" =>
      val ratings = sqlContext.read.text(dataPath + dataFileName).as[String]
        .map(x => {
          val line = x.split(delimiter).map(n => n.toInt)
          Rating(line(userIdx), line(itemIdx), line(labelIdx))
        })
        .filter(x => x.label > 0)
        .toDF()
      val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
      val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))
      (ratings, userCount, itemCount)

      case _ =>
      val ratings = sqlContext.read.text(dataPath + dataFileName).as[String]
        .map(x => {
          val line = x.split(delimiter).map(n => n.toInt)
          Rating(line(userIdx), line(itemIdx), line(labelIdx))
        })
        .filter(x => x.label > 0)
        .toDF()
      val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
      val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))
      (ratings, userCount, itemCount)

    }

  }

  // def loadPublicData(sqlContext: SQLContext, dataPath: String, dataFileName: String): (DataFrame, Int, Int) = {
  def loadPublicData(sqlContext: SQLContext, dataPath: String, fileName: String, delimiter: String): (DataFrame, Int, Int) = {
    import sqlContext.implicits._
    println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    /*
      change according to use:
      Data should be: int(userID), int(itemID), int
    */
    // val fileName = "/netflix_short.dat"
    println(s"fileName: $fileName. Delimiter: $delimiter")
    fileName match {
      case "/ratings.dat" =>
        readTextAndParse(sqlContext, dataPath, fileName, delimiter, 0, 1, 2) // T
      case "/netflix_short.dat" =>
        readTextAndParse(sqlContext, dataPath, fileName, delimiter, 1, 0, 2)
      case "/amazon_short.dat" =>
        readTextAndParse(sqlContext, dataPath, fileName, delimiter, 0, 1, 2) // T
      case "/lastfm_short.dat" =>
        readTextAndParse(sqlContext, dataPath, fileName, delimiter, 0, 1, 2)
      case "/book_short_orig.dat" =>
        readTextAndParse(sqlContext, dataPath, fileName, delimiter, 0, 1, 2)
      case _ =>
        readTextAndParse(sqlContext, dataPath, fileName, delimiter, 0, 1, 2)
    }

    // val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
    //   .map(x => {
    //     val line = x.split("::").map(n => n.toInt)
    //     Rating(line(0), line(1), line(2))
    //   }).toDF()

    // val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
    // val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))

    // (ratings, userCount, itemCount)
  }

  def assemblyFeature(isImplicit: Boolean = false,
                      indexed: DataFrame,
                      userCount: Int,
                      itemCount: Int): RDD[UserItemFeature[Float]] = {

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(indexed)
      negativeDF.unionAll(indexed.withColumn("label", lit(2)))
    }
    else indexed

    val rddOfSample: RDD[UserItemFeature[Float]] = unioned
      .select("userId", "itemId", "label")
      .rdd.map(row => {
      val uid = row.getAs[Int](0)
      val iid = row.getAs[Int](1)

      val label = row.getAs[Int](2)
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    rddOfSample
  }

}

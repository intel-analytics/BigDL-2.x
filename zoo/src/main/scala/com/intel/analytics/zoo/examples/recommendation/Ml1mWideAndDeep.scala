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

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Adam, Top1Accuracy}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.recommendation._
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.rdd.RDD

case class User(userId: Int, gender: String, age: Int, occupation: Int)

case class Item(itemId: Int, title: String, genres: String)

object Ml1mWideAndDeep {

  def run(params: WNDParams): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("WideAndDeepExample")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (ratingsDF, userDF, itemDF, userCount, itemCount) =
      loadPublicData(sqlContext, params.inputDir)

    ratingsDF.groupBy("label").count().show()
    val localColumnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("genderind", "ageind", "occupation", "genresfull", "genres1st"),
      wideBaseDims = Array(3, 8, 21, 500, 19),
      wideCrossCols = Array("gender-age", "gender-age-occupation", "gender-genres"),
      wideCrossDims = Array(500, 500, 500),
      indicatorCols = Array("genres1st", "genderind"),
      indicatorDims = Array(19, 3),
      embedCols = Array("userId", "itemId", "genres1st", "occupation"),
      embedInDims = Array(userCount, itemCount, 19, 21),
      embedOutDims = Array(64, 64, 64, 64),
      continuousCols = Array("age"))

    val wideAndDeep: WideAndDeep[Float] = WideAndDeep[Float](
      params.modelType,
      numClasses = 5,
      columnInfo = localColumnInfo)

    val isImplicit = false
    val featureRdds =
      assemblyFeature(isImplicit, ratingsDF, userDF, itemDF, localColumnInfo, params.modelType)

    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      featureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)

    val optimMethod = new Adam[Float](
      learningRate = 1e-3,
      learningRateDecay = 1e-6)

    wideAndDeep.compile(optimizer = optimMethod,
      loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      metrics = List(new Top1Accuracy[Float]())
    )
    wideAndDeep.fit(trainRdds, batchSize = params.batchSize,
      nbEpoch = params.maxEpoch, validationData = validationRdds)

    val results = wideAndDeep.predict(validationRdds)
    results.take(5).foreach(println)

    val resultsClass = wideAndDeep.predictClass(validationRdds)
    resultsClass.take(5).foreach(println)

    val userItemPairPrediction = wideAndDeep.predictUserItemPair(validationpairFeatureRdds)
    userItemPairPrediction.take(20).foreach(println)

    val userRecs = wideAndDeep.recommendForUser(validationpairFeatureRdds, 3)
    val itemRecs = wideAndDeep.recommendForItem(validationpairFeatureRdds, 3)

    userRecs.take(5).foreach(println)
    itemRecs.take(5).foreach(println)

    println("finished...")
    sc.stop()
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String):
  (DataFrame, DataFrame, DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/ratings.dat").as[String]
      .map(x => {
        val line = x.split("::").map(n => n.toInt)
        Rating(line(0), line(1), line(2))
      }).toDF()
    val userDF = sqlContext.read.text(dataPath + "/users.dat").as[String]
      .map(x => {
        val line = x.split("::")
        User(line(0).toInt, line(1), line(2).toInt, line(3).toInt)
      }).toDF()
    val itemDF = sqlContext.read.text(dataPath + "/movies.dat").as[String]
      .map(x => {
        val line = x.split("::")
        Item(line(0).toInt, line(1), line(2))
      }).toDF()

    val minMaxRow = ratings.agg(max("userId"), max("itemId")).collect()(0)
    val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))

    (ratings, userDF, itemDF, userCount, itemCount)
  }

  // convert features to RDD[Sample[Float]]
  def assemblyFeature(isImplicit: Boolean = false,
                      ratingDF: DataFrame,
                      userDF: DataFrame,
                      itemDF: DataFrame,
                      columnInfo: ColumnFeatureInfo,
                      modelType: String): RDD[UserItemFeature[Float]] = {

    val unioned = if (isImplicit) {
      val negativeDF = Utils.getNegativeSamples(ratingDF)
      negativeDF.unionAll(ratingDF.withColumn("label", lit(2)))
    }
    else ratingDF

    val genresList = Array("Crime", "Romance", "Thriller", "Adventure", "Drama", "Children's",
      "War", "Documentary", "Fantasy", "Mystery", "Musical", "Animation", "Film-Noir", "Horror",
      "Western", "Comedy", "Action", "Sci-Fi")
    val genresUDF = udf(Utils.categoricalFromVocabList(genresList))
    val genderUDF = udf(Utils.categoricalFromVocabList(Array("F", "M")))
    val ageUDF = udf(Utils.categoricalFromVocabList(Array("1", "18", "25", "35", "45", "50", "56")))
    val bucket1UDF = udf(Utils.buckBuckets(500)(_: String))
    val bucket2UDF = udf(Utils.buckBuckets(500)(_: String, _: String))
    val bucket3UDF = udf(Utils.buckBuckets(500)(_: String, _: String, _: String))

    val dataUse = unioned
      .join(itemDF, Array("itemId"))
      .join(userDF, Array("userId"))
      .withColumn("genderind", genderUDF(col("gender")))
      .withColumn("ageind", ageUDF(col("age")))
      .withColumn("gender-age", bucket2UDF(col("gender"), col("age")))
      .withColumn("gender-age-occupation", bucket3UDF(col("gender"), col("age"), col("occupation")))
      .withColumn("gender-genres", bucket2UDF(col("gender"), col("genres")))
      .withColumn("genres1st", genresUDF(col("genres")))
      .withColumn("genresfull", bucket1UDF(col("genres")))

    val rddOfSample = dataUse.rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      UserItemFeature(uid, iid, Utils.row2Sample(r, columnInfo, modelType))
    })
    rddOfSample
  }
}

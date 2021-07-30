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
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.dataset.Sample

case class MyUser(userId: Int, avgVote: Int)
case class MyItem(itemId: Int, category: String, price: Int)

object AmazonWideAndDeep {

  def run(params: WNDParams): Unit = {
    @transient lazy val logger = org.apache.log4j.LogManager.getLogger(
      "hscornelia")
      logger.info("AMAZOOOOOOOOON")
    val conf = new SparkConf().setAppName("AmazonWideAndDeepExample")
    val sc = NNContext.initNNContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)

    val (ratingsDF, userDF, itemDF, userCount, itemCount) =
      loadPublicData(sqlContext, params.inputDir)

    ratingsDF.groupBy("label").count().show()
    /*
      원래는 item -> (category), user -> (gender, age, occupation) 이었는데
      여기서는 item -> (category, price), user -> (avgVote) avgVote는 자신이 쓴 리뷰가 받은 vote의 평균
      원래 dimension은: category: 29개
                      price: ?? 실제 데이터에서 나온 가격 보고 나눈 다음에 정해야할것 같음
                      avgVote: ?? price와 같은 상황

      작은 인풋: 작은 인풋을 돌리는 목적이 코드가 잘 돌아가는지 확인용이고, 위에서 price, avgVote 같이 아직 데이터가
      완성되지 않아서 카테고리가 미정인 feature들도 있어서 그냥 임의로 만듬...
      category: 2 dimension (Magazine, Clothes)
      price: 3 dim (1: [0, 5), 5: [5, 10), 10: [10, 15))
      avgVote: 3 dim (1: [0, 5), 5: [5, 10), 10: [10, 15))
    */
    val localColumnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("categoryind", "priceind", "avgVoteind"), // actual (2, 3, 3) -- to ind --> (3, 4, 4)
      wideBaseDims = Array(3, 4, 4),
      wideCrossCols = Array("category-price"),
      wideCrossDims = Array(100),
      indicatorCols = Array("categoryind"),
      indicatorDims = Array(3),
      embedCols = Array("userId", "itemId", "avgVoteind"),
      embedInDims = Array(userCount, itemCount, 4),
      embedOutDims = Array(64, 64, 64),
      continuousCols = Array("priceind")
    )

    val wideAndDeep: WideAndDeep[Float] = WideAndDeep[Float](
      params.modelType,
      numClasses = 5,
      columnInfo = localColumnInfo)

    val isImplicit = false
    val featureRdds =
      assemblyFeature(isImplicit, ratingsDF, userDF, itemDF, localColumnInfo, params.modelType)

    println("Feature RDD")
    println("Blablabla Features")


    val Array(trainpairFeatureRdds, validationpairFeatureRdds) =
      featureRdds.randomSplit(Array(0.8, 0.2))
    val trainRdds = trainpairFeatureRdds.map(x => x.sample)
    val validationRdds = validationpairFeatureRdds.map(x => x.sample)

    println("Train RDD")
    trainRdds.collect().foreach(println)

    println("Validation RDD")
    validationRdds.collect().foreach(println)

    val optimMethod = new Adam[Float](
      learningRate = 1e-3,
      learningRateDecay = 1e-6)

    wideAndDeep.compile(optimizer = optimMethod,
      loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      metrics = List(new Top1Accuracy[Float]())
    )
    wideAndDeep.fit(trainRdds, batchSize = params.batchSize,
      nbEpoch = params.maxEpoch, validationData = validationRdds)

    println("finished...")
    sc.stop()
  }

  def loadPublicData(sqlContext: SQLContext, dataPath: String):
  (DataFrame, DataFrame, DataFrame, Int, Int) = {
    import sqlContext.implicits._
    val ratings = sqlContext.read.text(dataPath + "/amazon_short.dat").as[String]
      .map(x => {
        val line = x.split(":").map(n => n.toInt)
        Rating(line(0), line(1), line(2))
      }).toDF()
    val userDF = sqlContext.read.text(dataPath + "/amazon_user_short.dat").as[String]
      .map(x => {
        val line = x.split(":")
        MyUser(line(0).toInt, line(1).toInt)
      }).toDF()
    val itemDF = sqlContext.read.text(dataPath + "/amazon_item_short.dat").as[String]
      .map(x => {
        val line = x.split(":")
        MyItem(line(0).toInt, line(1), line(2).toInt)
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

    val categoryList = Array("Magazine", "Clothing")
    val categoryUDF = udf(Utils.categoricalFromVocabList(categoryList))
    val priceUDF = udf(Utils.categoricalFromVocabList(Array("1", "5", "10")))
    val avgVoteUDF = udf(Utils.categoricalFromVocabList(Array("1", "5", "10")))
    val bucket1UDF = udf(Utils.buckBuckets(100)(_: String))
    val bucket2UDF = udf(Utils.buckBuckets(100)(_: String, _: String))
    val bucket3UDF = udf(Utils.buckBuckets(100)(_: String, _: String, _: String))

    val dataUse = unioned
      .join(itemDF, Array("itemId"))
      .join(userDF, Array("userId"))
      .withColumn("priceind", priceUDF(col("price")))
      .withColumn("categoryind", categoryUDF(col("category")))
      .withColumn("category-price", bucket2UDF(col("category"), col("price")))
      .withColumn("avgVoteind", avgVoteUDF(col("avgVote")))
      .withColumn("categoryfull", bucket1UDF(col("category")))
    unioned.show()
    itemDF.show()
    userDF.show()
    dataUse.show()
    dataUse.printSchema()
    val rddOfSample = dataUse.rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      UserItemFeature(uid, iid, Utils.row2Sample(r, columnInfo, modelType))
    })
    rddOfSample
  }
}

  /* Sample
+------+------+-----+--------+-----+-------+--------+-----------+--------------+----------+------------+
|userId|itemId|label|category|price|avgVote|priceind|categoryind|category-price|avgVoteind|categoryfull|
+------+------+-----+--------+-----+-------+--------+-----------+--------------+----------+------------+
|     0|     0|    5|Clothing|    1|      1|       1|          2|            22|         1|          72|
|     0|     1|    5|Magazine|    5|      1|       2|          1|            34|         1|          32|
|     0|     2|    5|Clothing|    1|      1|       1|          2|            22|         1|          72|
|     0|     3|    4|Magazine|    5|      1|       2|          1|            34|         1|          32|
|     0|     4|    5|Magazine|    1|      1|       1|          1|            30|         1|          32|
|     0|     5|    4|Magazine|    5|      1|       2|          1|            34|         1|          32|
|     0|     6|    5|Clothing|    1|      1|       1|          2|            22|         1|          72|
|     1|     7|    5|Clothing|   10|      5|       3|          2|            42|         2|          72|
|     1|     8|    5|Magazine|    5|      5|       2|          1|            34|         2|          32|
|     1|     9|    3|Clothing|   10|      5|       3|          2|            42|         2|          72|
+------+------+-----+--------+-----+-------+--------+-----------+--------------+----------+------------+
*/

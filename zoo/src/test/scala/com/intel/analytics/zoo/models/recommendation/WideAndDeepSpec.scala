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

package com.intel.analytics.zoo.models.recommendation

import java.net.URL

import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions._

import scala.util.Random

class WideAndDeepSpec extends ZooSpecHelper {

  var sqlContext: SQLContext = _
  var sc: SparkContext = _
  var datain: DataFrame = _
  val categoricalUDF = udf(Utils.categoricalFromVocabList(Array("F", "M")))
  val bucketUDF = udf(Utils.buckBucket(100))

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("WideNdeepTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)

    val resource: URL = getClass.getClassLoader.getResource("recommender")
    datain = sqlContext.read.parquet(resource.getFile)
      .withColumn("gender", categoricalUDF(col("gender")))
      .withColumn("occupation-gender", bucketUDF(col("occupation"), col("gender")))
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "WideAndDeep wide model forward and backward" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("occupation", "gender"),
      wideBaseDims = Array(21, 3),
      wideCrossCols = Array("occupation-gender"),
      wideCrossDims = Array(100))
    val model = WideAndDeep[Float]("wide", 5, columnInfo)

    val data = datain
      .rdd.map(r => Utils.getWideTensor(r, columnInfo))
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }.count()

  }


  "WideAndDeep deep model indicator forward and backward" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      indicatorCols = Array("occupation", "gender"),
      indicatorDims = Array(21, 3))
    val model = WideAndDeep[Float]("deep", 5, columnInfo)

    val data = datain
      .rdd.map(r => Utils.getDeepTensor(r, columnInfo))
    data.map{ input =>
      val feature : Tensor[Float] = input.reshape(Array(1, input.size(1)))
      val output = model.forward(feature)
      val gradInput = model.backward(feature, output)
    }.count()
  }

  "WideAndDeep deep model embedding and continuous part" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      indicatorCols = Array("occupation", "gender"),
      indicatorDims = Array(21, 3),
      embedCols = Array("userId", "itemId"),
      embedInDims = Array(100, 100),
      embedOutDims = Array(20, 20),
      continuousCols = Array("age")
    )
    val model = WideAndDeep[Float]("deep", 5, columnInfo)

    val data: RDD[Tensor[Float]] = datain
      .rdd.map(r => Utils.getDeepTensor(r, columnInfo))

    data.map{ input =>
      val feature : Tensor[Float] = input.reshape(Array(1, input.size(1)))
      val output = model.forward(feature)
      val gradInput = model.backward(feature, output)
    }.count()
  }

  "WideAndDeep full model forward and backward" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("occupation", "gender"),
      wideBaseDims = Array(21, 3),
      wideCrossCols = Array("occupation-gender"),
      wideCrossDims = Array(100),
      indicatorCols = Array("occupation", "gender"),
      indicatorDims = Array(21, 3),
      embedCols = Array("userId", "itemId"),
      embedInDims = Array(100, 100),
      embedOutDims = Array(20, 20),
      continuousCols = Array("age"))
    val model = WideAndDeep[Float]("wide_n_deep", 5, columnInfo)

    val data: RDD[Activity] = datain
      .rdd.map(r => {
      val wideTensor: Tensor[Float] = Utils.getWideTensor(r, columnInfo)
      val deepTensor: Tensor[Float] = Utils.getDeepTensor(r, columnInfo)
      T(T(wideTensor, deepTensor))
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }.count()
  }

  "WideAndDeep full model implicit forward and backward" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("occupation", "gender"),
      wideBaseDims = Array(21, 3),
      wideCrossCols = Array("occupation-gender"),
      wideCrossDims = Array(100),
      indicatorCols = Array("occupation", "gender"),
      indicatorDims = Array(21, 3),
      embedCols = Array("userId", "itemId"),
      embedInDims = Array(100, 100),
      embedOutDims = Array(20, 20),
      continuousCols = Array("age"))
    val model = WideAndDeep[Float]("wide_n_deep", 2, columnInfo)

    val negativeDF = Utils.getNegativeSamples(datain)
    val userDF = datain.select("userId", "occupation", "gender", "age")

    val unioned = negativeDF.join(userDF, Array("userId"))
      .select("occupation", "gender", "age", "userId", "itemId")
      .unionAll(datain.select("occupation", "gender", "age", "userId", "itemId"))
      .withColumn("occupation-gender", bucketUDF(col("occupation"), col("gender")))

    val data: RDD[Activity] = unioned
      .rdd.map(r => {
      val wideTensor: Tensor[Float] = Utils.getWideTensor(r, columnInfo)
      val deepTensor: Tensor[Float] = Utils.getDeepTensor(r, columnInfo)
      T(T(wideTensor, deepTensor))
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }.count()
  }

  "WideAndDeep predictUserItemPair" should "have correct predictions" in {
    val columnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("occupation", "gender"),
      wideBaseDims = Array(21, 3),
      wideCrossCols = Array("occupation-gender"),
      wideCrossDims = Array(100),
      indicatorCols = Array("occupation", "gender"),
      indicatorDims = Array(21, 3),
      embedCols = Array("userId", "itemId"),
      embedInDims = Array(100, 100),
      embedOutDims = Array(20, 20),
      continuousCols = Array("age"))
    val widendeep = WideAndDeep[Float]("wide_n_deep", 5, columnInfo)

    val data = datain
      .withColumn("occupation-gender", bucketUDF(col("occupation"), col("gender")))
      .rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      UserItemFeature(uid, iid, Utils.row2Sample(r, columnInfo, "wide_n_deep"))
    })
    val trainRdds = data.map(x => x.sample)

    val optimizer = Optimizer(
      model = widendeep,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 458)

    optimizer
      .setOptimMethod(new Adam[Float](learningRate = 1e-2, learningRateDecay = 1e-5))
      .setEndWhen(Trigger.maxEpoch(100))
      .optimize()

    val pairPredictions = widendeep.predictUserItemPair(data)
    val pairPredictionsDF = sqlContext.createDataFrame(pairPredictions).toDF()
    val out = pairPredictionsDF.join(datain, Array("userId", "itemId"))
    val correctCounts = out.filter(col("prediction") === col("label")).count()

    val accuracy = correctCounts.toDouble / 458
    // the reference accuracy in this testcase is 0.856
    assert(accuracy >= 0.40)
  }

  "WideAndDeep recommendForItem and recommendForItem" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      wideBaseCols = Array("occupation", "gender"),
      wideBaseDims = Array(21, 3),
      wideCrossCols = Array("occupation-gender"),
      wideCrossDims = Array(100),
      indicatorCols = Array("occupation", "gender"),
      indicatorDims = Array(21, 3),
      embedCols = Array("userId", "itemId"),
      embedInDims = Array(100, 100),
      embedOutDims = Array(20, 20),
      continuousCols = Array("age"))
    val widendeep = WideAndDeep[Float]("wide_n_deep", 5, columnInfo)

    val data = datain
      .withColumn("occupation-gender", bucketUDF(col("occupation"), col("gender")))
      .rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      UserItemFeature(uid, iid, Utils.row2Sample(r, columnInfo, "wide_n_deep"))
    })
    val trainRdds = data.map(x => x.sample)

    val optimizer = Optimizer(
      model = widendeep,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 458)

    optimizer
      .setOptimMethod(new Adam[Float](learningRate = 1e-2, learningRateDecay = 1e-5))
      .setEndWhen(Trigger.maxEpoch(100))
      .optimize()

    val itemRecs = widendeep.recommendForItem(data, 2)
    val userRecs = widendeep.recommendForUser(data, 2)

    itemRecs.map(x => x.probability).sample(false, 0.1, 1L).collect()
      .map(x => assert(x >= 0.2))
    userRecs.map(x => x.probability).sample(false, 0.1, 1L).collect()
      .map(x => assert(x >= 0.2))

    assert(itemRecs.count() <= 200)
    assert(userRecs.count() <= 200)
  }

}

class WideAndDeepSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val columnInfo = ColumnFeatureInfo(
      indicatorCols = Array("occupation", "gender"),
      indicatorDims = Array(21, 3))
    val model = WideAndDeep[Float]("deep", 5, columnInfo)
    val input = Tensor[Float](Array(100, 50))
      .fill(new Random(System.nanoTime()).nextInt(20).toFloat + 1)
    ZooSpecHelper.testZooModelLoadSave(model, input, WideAndDeep.loadModel[Float])
  }
}

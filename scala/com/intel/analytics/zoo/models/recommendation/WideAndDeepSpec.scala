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

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{RandomGenerator, T, Table}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions._


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
    data.take(10).map { x =>
      val input = x.resize(Array(1, 124), 3)
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "WideAndDeep indicator forward and backward" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      indicatorCols = Array("occupation", "gender"),
      indicatorDims = Array(21, 3))
    val model = WideAndDeep[Float]("deep", 5, columnInfo)

    val data = datain
      .rdd.map(r => Utils.row2Sample(r, columnInfo, "deep").feature())
    data.map { input =>
      val feature: Tensor[Float] = input.reshape(Array(1, input.size(1)))
      val output = model.forward(feature)
      val gradInput = model.backward(feature, output)
    }.count()
  }

  "WideAndDeep embedding forward and backward" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      embedCols = Array("userId", "itemId"),
      embedInDims = Array(100, 100),
      embedOutDims = Array(20, 20))

    val model = WideAndDeep[Float]("deep", 5, columnInfo)
    val data = datain
      .rdd.map(r => Utils.row2Sample(r, columnInfo, "deep").feature())

    data.map { input =>
      val feature: Tensor[Float] = input.reshape(Array(1, input.size(1)))
      val output = model.forward(feature)
      val gradInput = model.backward(feature, output)
    }.count()
  }

  "WideAndDeep continuous forward and backward" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      continuousCols = Array("age")
    )
    val model = WideAndDeep[Float]("deep", 5, columnInfo)

    val data = datain
      .rdd.map(r =>
      Utils.row2Sample(r, columnInfo, "deep").feature())

    data.map { input =>
      val feature: Tensor[Float] = input.reshape(Array(1, input.size(1)))
      val output = model.forward(feature)
      val gradInput = model.backward(feature, output)
    }.count()
  }

  "WideAndDeep deep model forward and backward" should "work properly" in {
    val columnInfo = ColumnFeatureInfo(
      indicatorCols = Array("occupation", "gender"),
      indicatorDims = Array(21, 3),
      embedCols = Array("userId", "itemId"),
      embedInDims = Array(100, 100),
      embedOutDims = Array(20, 20),
      continuousCols = Array("age")
    )
    val model = WideAndDeep[Float]("deep", 5, columnInfo)

    val data = datain
      .rdd.map(r => Utils.getDeepTensors(r, columnInfo))

    data.map { input =>
      val feature: Table = T.array(input.map(x => x.reshape(Array(1, x.size(1)))))
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
      val wideTensor: Tensor[Float] = Utils.getWideTensor(r, columnInfo).resize(Array(1, 124), 3)
      val deepTensor: Array[Tensor[Float]] = Utils.getDeepTensors(r, columnInfo)
          .map(x => x.reshape(Array(1, x.size(1))))
      T.array(Array(wideTensor) ++ deepTensor)
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }.count()
  }

    "WideAndDeep wide fit and compile" should "work properly" in {

      val columnInfo = ColumnFeatureInfo(
        wideBaseCols = Array("occupation", "gender"),
        wideBaseDims = Array(21, 3),
        wideCrossCols = Array("occupation-gender"),
        wideCrossDims = Array(100))
      val widendeep = WideAndDeep[Float]("wide", 5, columnInfo)

      val data = datain
        .withColumn("occupation-gender", bucketUDF(col("occupation"), col("gender")))
        .rdd.map(r => {
        val uid = r.getAs[Int]("userId")
        val iid = r.getAs[Int]("itemId")
        UserItemFeature(uid, iid, Utils.row2Sample(r, columnInfo, "wide"))
      })
      val trainRdds = data.map(x => x.sample)

      val optimMethod = new Adam[Float](
        learningRate = 1e-2,
        learningRateDecay = 1e-5)

      widendeep.compile(optimizer = optimMethod,
        loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
        metrics = List(new Top1Accuracy[Float]())
      )
      widendeep.fit(trainRdds, batchSize = 458,
        nbEpoch = 1, validationData = trainRdds)
    }

    "WideAndDeep fit and compile" should " work properly" in {
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

      val optimMethod = new Adam[Float](
        learningRate = 1e-2,
        learningRateDecay = 1e-5)

      widendeep.compile(optimizer = optimMethod,
        loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
        metrics = List(new Top1Accuracy[Float]())
      )
      widendeep.fit(trainRdds, batchSize = 458,
        nbEpoch = 20, validationData = trainRdds)

      val pairPredictions = widendeep.predictUserItemPair(data)
      val pairPredictionsDF = sqlContext.createDataFrame(pairPredictions).toDF()
      val out = pairPredictionsDF.join(datain, Array("userId", "itemId"))
      val correctCounts = out.filter(col("prediction") === col("label")).count()

      val accuracy = correctCounts.toDouble / 458
      println(accuracy)
      // the reference accuracy in this testcase is 0.858
      assert(accuracy >= 0.30)
    }

    "WideAndDeep recommendForItem and recommendForUser" should "work properly" in {
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

      val optimMethod = new Adam[Float](
        learningRate = 1e-2,
        learningRateDecay = 1e-5)

      widendeep.compile(optimizer = optimMethod,
        loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
        metrics = List(new Top1Accuracy[Float]())
      )
      widendeep.fit(trainRdds, batchSize = 458,
        nbEpoch = 20, validationData = trainRdds)

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
      embedCols = Array("occupation", "gender"),
      embedInDims = Array(21, 21),
      embedOutDims = Array(4, 4))
    val model = WideAndDeep[Float]("deep", 5, columnInfo)
    val rng = RandomGenerator.RNG
    val input = Tensor[Float](Array(100, 2))
      .apply1(_ => rng.uniform(1, 21).toInt)
    ZooSpecHelper.testZooModelLoadSave(model, input, WideAndDeep.loadModel[Float])
  }
}

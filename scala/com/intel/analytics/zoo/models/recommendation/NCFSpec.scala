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

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.util.Random

class NCFSpec extends ZooSpecHelper {

  var sqlContext: SQLContext = _
  var sc: SparkContext = _
  var datain: DataFrame = _
  val userCount = 100
  val itemCount = 100

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("NCFTest")
    sc = NNContext.getNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)

    datain = sqlContext.read.parquet("./src/test/resources/recommender/")
      .select("userId", "itemId", "label")
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "NeuralCF without MF forward and backward" should "work properly" in {

    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), false)
    val ran = new Random(42L)
    val data: Seq[Tensor[Float]] = (1 to 50).map(i => {
      val uid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val iid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val feature: Tensor[Float] = Tensor(T(T(uid, iid)))
      println(feature.size().toList)
      val label = Math.abs(ran.nextInt(4)).toFloat + 1
      feature
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "NeuralCF with MF forward and backward" should "work properly" in {
    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), true, 3)
    val ran = new Random(42L)
    val data: Seq[Tensor[Float]] = (1 to 50).map(i => {
      val uid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val iid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val feature: Tensor[Float] = Tensor(T(T(uid, iid)))
      val label = Math.abs(ran.nextInt(4)).toFloat + 1
      feature
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "NeuralCF predictUserItemPair" should "have correct predictions" in {

    val ncf = NeuralCF[Float](100, 100, 5, 5, 5, Array(10, 5), false)
    val data = datain
      .rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      val label = r.getAs[Int]("label")
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    val trainRdds = data.map(x => x.sample)

    val optimizer = Optimizer(
      model = ncf,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 458)

    optimizer
      .setOptimMethod(new Adam[Float](learningRate = 1e-2, learningRateDecay = 1e-5))
      .setEndWhen(Trigger.maxEpoch(100))
      .optimize()

    val pairPredictions = ncf.predictUserItemPair(data)
    val pairPredictionsDF = sqlContext.createDataFrame(pairPredictions).toDF()
    val out = pairPredictionsDF.join(datain, Array("userId", "itemId"))
    val correctCounts = out.filter(col("prediction") === col("label")).count()

    val accuracy = correctCounts.toDouble / 458
    // the reference accuracy is 0.679
    assert(accuracy >= 0.40)
  }

  "NeuralCF recommendForItem and recommendForItem" should "have correct predictions" in {

    val ncf = NeuralCF[Float](100, 100, 5, 5, 5, Array(10, 5), false)
    val data = datain
      .rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      val label = r.getAs[Int]("label")
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    val trainRdds = data.map(x => x.sample)

    val optimizer = Optimizer(
      model = ncf,
      sampleRDD = trainRdds,
      criterion = ClassNLLCriterion[Float](),
      batchSize = 458)

    optimizer
      .setOptimMethod(new Adam[Float](learningRate = 1e-2, learningRateDecay = 1e-5))
      .setEndWhen(Trigger.maxEpoch(100))
      .optimize()

    val itemRecs = ncf.recommendForItem(data, 2)
    val userRecs = ncf.recommendForUser(data, 2)

    itemRecs.map(x => x.probability).sample(false, 0.1, 1L).collect()
      .map(x => assert(x >= 0.2))
    userRecs.map(x => x.probability).sample(false, 0.1, 1L).collect()
      .map(x => assert(x >= 0.2))

    assert(itemRecs.count() <= 200)
    assert(userRecs.count() <= 200)
  }

  "NeuralCF save and load" should "work properly" in {
    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), false)
    val input = Tensor[Float](Array(100, 2))
      .fill(new Random(System.nanoTime()).nextInt(userCount - 1).toFloat + 1)
    testZooModelLoadSave(model, input, NeuralCF.loadModel[Float])
  }
}

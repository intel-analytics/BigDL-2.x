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

import com.intel.analytics.bigdl.dataset.{Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, MSECriterion}
import com.intel.analytics.bigdl.optim.{Adam, LBFGS, Optimizer, Top1Accuracy, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.models.python.PythonZooModel
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.models.KerasNet
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import com.intel.analytics.zoo.pipeline.api.keras.python.PythonZooKeras
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.estimator.Estimator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.util.Random

class NeuralCFSpec extends ZooSpecHelper {

  var sqlContext: SQLContext = _
  var sc: SparkContext = _
  var datain: DataFrame = _
  val userCount = 100
  val itemCount = 100

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("NCFTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)
    val resource: URL = getClass.getClassLoader.getResource("recommender")
    datain = sqlContext.read.parquet(resource.getFile)
      .select("userId", "itemId", "label")
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "NeuralCF without MF forward and backward" should "work properly" in {

    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), false)
    model.summary()
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

    val optimMethod = new Adam[Float](learningRate = 1e-2, learningRateDecay = 1e-5)

    val sample2batch = SampleToMiniBatch[Float](458)
    val trainSet = FeatureSet.rdd(trainRdds.cache()) -> sample2batch

    val estimator = Estimator[Float](ncf, optimMethod)

    val (checkpointTrigger, endTrigger) =
      (Trigger.everyEpoch, Trigger.maxEpoch(100))

    estimator.train(trainSet, SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      Some(endTrigger), Some(checkpointTrigger), null, Array(new Top1Accuracy[Float]()))

    val pairPredictions = ncf.predictUserItemPair(data)
    val pairPredictionsDF = sqlContext.createDataFrame(pairPredictions).toDF()
    val out = pairPredictionsDF.join(datain, Array("userId", "itemId"))
    val correctCounts = out.filter(col("prediction") === col("label")).count()

    val accuracy = correctCounts.toDouble / 458
    // the reference accuracy is 0.679
    assert(accuracy >= 0.40)
  }

  "NeuralCF recommendForItem and recommendForUser" should "have correct predictions" in {

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

    val optimMethod = new Adam[Float](learningRate = 1e-2, learningRateDecay = 1e-5)

    val sample2batch = SampleToMiniBatch[Float](458)
    val trainSet = FeatureSet.rdd(trainRdds.cache()) -> sample2batch

    val estimator = Estimator[Float](ncf, optimMethod)

    val (checkpointTrigger, endTrigger) =
      (Trigger.everyEpoch, Trigger.maxEpoch(100))

    estimator.train(trainSet, SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      Some(endTrigger), Some(checkpointTrigger), null, Array(new Top1Accuracy[Float]()))

    val itemRecs = ncf.recommendForItem(data, 2)
    val userRecs = ncf.recommendForUser(data, 2)

    itemRecs.map(x => x.probability).sample(false, 0.1, 1L).collect()
      .map(x => assert(x >= 0.2))
    userRecs.map(x => x.probability).sample(false, 0.1, 1L).collect()
      .map(x => assert(x >= 0.2))

    assert(itemRecs.count() <= 200)
    assert(userRecs.count() <= 200)
  }

  "NeuralCF compile and fit" should "has similar performance to Estimator" in {
    val data = datain
      .rdd.map(r => {
      val uid = r.getAs[Int]("userId")
      val iid = r.getAs[Int]("itemId")
      val label = r.getAs[Int]("label")
      val feature: Tensor[Float] = Tensor[Float](T(uid.toFloat, iid.toFloat))

      UserItemFeature(uid, iid, Sample(feature, Tensor[Float](T(label))))
    })
    val trainRdds = data.map(x => x.sample)



    // Use Estimator API
    val ncfEst = NeuralCF[Float](100, 100, 5, 5, 5, Array(10, 5), false)
    val ncf = ncfEst.cloneModule()
    val sample2batch = SampleToMiniBatch[Float](458)
    val trainSet = FeatureSet.rdd(trainRdds.cache()) -> sample2batch
    val estOptimMethod = new Adam[Float](
      learningRate = 1e-2,
      learningRateDecay = 1e-5)
    val estimator = Estimator[Float](ncfEst, estOptimMethod)

    val (checkpointTrigger, endTrigger) =
      (Trigger.everyEpoch, Trigger.maxEpoch(100))

    estimator.train(trainSet, SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      Some(endTrigger), Some(checkpointTrigger), null, Array(new Top1Accuracy[Float]()))

    val pairPredictionsEst = ncfEst.predictUserItemPair(data)
    val pairPredictionsEstDF = sqlContext.createDataFrame(pairPredictionsEst).toDF()
    val outEst = pairPredictionsEstDF.join(datain, Array("userId", "itemId"))
    val correctCountsEst = outEst.filter(col("prediction") === col("label")).count()

    val accuracyEst = correctCountsEst.toDouble / 458

    // Use compile and fit API
    val optimMethod = new Adam[Float](
      learningRate = 1e-2,
      learningRateDecay = 1e-5)
    ncf.compile(optimizer = optimMethod,
      loss = SparseCategoricalCrossEntropy[Float](zeroBasedLabel = false),
      metrics = List(new Top1Accuracy[Float]()))

    ncf.fit(trainRdds, batchSize = 458, nbEpoch = 100, validationData = null)

    val pairPredictions = ncf.predictUserItemPair(data)
    val pairPredictionsDF = sqlContext.createDataFrame(pairPredictions).toDF()
    val out = pairPredictionsDF.join(datain, Array("userId", "itemId"))
    val correctCounts = out.filter(col("prediction") === col("label")).count()

    val accuracy = correctCounts.toDouble / 458

    // the reference accuracy is 0.679
    assert(Math.abs(accuracy - accuracyEst) < 0.1)
  }


}

class NeuralCFSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = NeuralCF[Float](100, 100, 5, 5, 5, Array(10, 5), false)
    val input = Tensor[Float](Array(100, 2))
      .fill(new Random(System.nanoTime()).nextInt(100 - 1).toFloat + 1)
    ZooSpecHelper.testZooModelLoadSave(model, input, NeuralCF.loadModel[Float])
  }
}

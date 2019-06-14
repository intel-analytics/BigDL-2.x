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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.anomalydetection.AnomalyDetector
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

class SessionRecommenderSpec extends ZooSpecHelper {

  var sqlContext: SQLContext = _
  var sc: SparkContext = _

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("NCFTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "SessionRecommender without history forward and backward" should "work properly" in {

    val itemCount = 100
    val sessionLength = 10
    val model = SessionRecommender[Float](itemCount, sessionLength, includeHistory = false)
    val ran = new Random(42L)
    val data = (1 to 100).map { x =>
      val items: Seq[Float] = for (i <- 1 to sessionLength) yield
        ran.nextInt(itemCount - 1).toFloat + 1
      Tensor(items.toArray, Array(sessionLength)).resize(1, sessionLength)
    }
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "SessionRecommender with history forward and backward" should "work properly" in {
    val itemCount = 100
    val sessionLength = 10
    val historyLength = 5
    val model = SessionRecommender[Float](itemCount, sessionLength,
      includeHistory = true, historyLength = historyLength)
    val ran = new Random(42L)
    val data = (1 to 100).map { x =>
      val items1: Seq[Float] = for (i <- 1 to sessionLength) yield
        ran.nextInt(itemCount - 1).toFloat + 1
      val items2: Seq[Float] = for (i <- 1 to historyLength) yield
        ran.nextInt(itemCount - 1).toFloat + 1
      val input1 = Tensor(items1.toArray, Array(sessionLength)).resize(1, sessionLength)
      val input2 = Tensor(items2.toArray, Array(historyLength)).resize(1, historyLength)
      T(input1, input2)
    }
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "SessionRecommender recommendForSession" should "work properly" in {
    val itemCount = 100
    val sessionLength = 10
    val historyLength = 5
    val model = SessionRecommender[Float](itemCount, sessionLength,
      includeHistory = true, historyLength = historyLength)
    val ran = new Random(42L)
    val data1: RDD[Sample[Float]] = sc.parallelize(1 to 100)
      .map { x =>
        val items1: Seq[Float] = for (i <- 1 to sessionLength) yield ran.nextInt(itemCount).toFloat
        val items2: Seq[Float] = for (i <- 1 to historyLength) yield ran.nextInt(itemCount).toFloat
        val input1 = Tensor(items1.toArray, Array(sessionLength))
        val input2 = Tensor(items2.toArray, Array(historyLength))
        Sample[Float](Array(input1, input2))
      }

    val recommedations1 = model.recommendForSession(data1, 3, zeroBasedLabel = false)
    recommedations1.take(10)
      .map { x =>
        assert(x.size == 3)
        assert(x(0)._2 >= x(1)._2)
      }

    val data2: Array[Sample[Float]] = (1 to 10)
      .map { x =>
        val items1: Seq[Float] = for (i <- 1 to sessionLength) yield ran.nextInt(itemCount).toFloat
        val items2: Seq[Float] = for (i <- 1 to historyLength) yield ran.nextInt(itemCount).toFloat
        val input1 = Tensor(items1.toArray, Array(sessionLength))
        val input2 = Tensor(items2.toArray, Array(historyLength))
        Sample[Float](Array(input1, input2))
      }.toArray

    val recommedations2 = model.recommendForSession(data2, 4, zeroBasedLabel = false)
    recommedations2.map { x =>
      assert(x.size == 4)
      assert(x(0)._2 >= x(1)._2)
    }
  }

  "SessionRecommender compile and fit" should "work properly" in {

    val itemCount = 100
    val sessionLength = 10
    val historyLength = 5
    val model = SessionRecommender[Float](itemCount, sessionLength,
      includeHistory = true, historyLength = historyLength)
    val ran = new Random(42L)
    val data1 = sc.parallelize(1 to 100)
      .map { x =>
        val items1: Seq[Float] = for (i <- 1 to sessionLength) yield ran.nextInt(itemCount).toFloat
        val items2: Seq[Float] = for (i <- 1 to historyLength) yield ran.nextInt(itemCount).toFloat
        val input1 = Tensor(items1.toArray, Array(sessionLength))
        val input2 = Tensor(items2.toArray, Array(historyLength))
        val label = Tensor[Float](T(ran.nextInt(itemCount).toFloat))
        Sample(Array(input1, input2), Array(label))
      }
    model.compile(optimizer = "rmsprop", loss = "sparse_categorical_crossentropy")
    model.fit(data1, nbEpoch = 1)
  }
}

class SessionRecommenderSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val ran = new Random(42L)
    val itemCount = 100
    val sessionLength = 10
    val model = SessionRecommender[Float](100, 10, includeHistory = false)
    val items: Seq[Float] = for (i <- 1 to sessionLength) yield
      ran.nextInt(itemCount - 1).toFloat + 1
    val data = Tensor(items.toArray, Array(sessionLength)).resize(1, sessionLength)
    ZooSpecHelper.testZooModelLoadSave(model, data, SessionRecommender.loadModel[Float])
  }
}

/*
 * Copyright 2021 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.narwhal.fl.vertical.nn

import java.util

import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.bigdl.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.zoo.ppml.FLClient
import com.intel.analytics.zoo.ppml.vfl.VflEstimator
import com.intel.analytics.zoo.ppml.psi.test.TestUtils
import com.intel.analytics.zoo.ppml.vfl.utils.SampleToMiniBatch
import org.apache.log4j.Logger

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.io.Source

object VflLogisticRegression {
  val logger = Logger.getLogger(this.getClass)
  val flClient = new FLClient()
  protected var hashedKeyPairs: Map[String, String] = null
  def main(args: Array[String]): Unit = {
    // load args
    val datapath = args(0)
    val worker = args(1).toInt
    val batchSize = args(2).toInt
    val learningRate = args(3).toFloat
    val rowkeyName = args(4)

    // load data from dataset and preprocess
    val sources = Source.fromFile(datapath, "utf-8").getLines()
    val headers = sources.next().split(",").map(_.trim)
    println(headers.mkString(","))
    val rowKeyIndex = headers.indexOf(rowkeyName)
    require(rowKeyIndex != -1, s"couldn't find ${rowkeyName} in headers(${headers.mkString(", ")})")
    val data = sources.toArray.map{line =>
      val lines = line.split(",").map(_.trim())
      (lines(rowKeyIndex), (lines.take(rowKeyIndex) ++ lines.drop(rowKeyIndex + 1)).map(_.toFloat))
    }.toMap
    val ids = data.keys.toArray
    uploadKeys(ids)
    val intersections = getIntersectionKeys()
    val trainData = intersections.map{id =>
      data(id)
    }

    val (samples, featureNum) = if (headers.last == "Outcome") {
      println("hasLabel")
      val featureNum = headers.length - 2
      (0 until featureNum).foreach(i => minMaxNormalize(trainData, i))
      (trainData.map{d =>
        val features = Tensor[Float](d.slice(0, featureNum), Array(featureNum))
        val target = Tensor[Float](Array(d(featureNum)), Array(1))
        Sample(features, target)
      }, featureNum)
    } else {
      println("no label")
      val featureNum = headers.length - 1
      (0 until featureNum).foreach(i => minMaxNormalize(trainData, i))
      (trainData.map{d =>
        val features = Tensor[Float](d, Array(featureNum))
        Sample(features)
      }, featureNum)
    }
    val trainDataset = DataSet.array(samples) -> SampleToMiniBatch(batchSize)
    //TODO: Find a better dataset has val dataset.
    val valDataset = DataSet.array(samples) -> SampleToMiniBatch(batchSize)
    // define model
    RNG.setSeed(worker)
    val model = if (worker == 0) {
      Sequential[Float]().add(Linear(featureNum, 1))
    } else {
      Sequential[Float]().add(Linear(featureNum, 1, withBias = false))
    }
    println(model.getParametersTable())
    val estimator = VflEstimator(model, new Adam(learningRate))
    estimator.train(30, trainDataset.toLocal(), valDataset.toLocal())
    // check training result
    println(model.getParametersTable())
    estimator.getEvaluateResults().foreach{r =>
      println(r._1 + ":" + r._2.mkString(","))
    }
  }

  def minMaxNormalize(data: Array[Array[Float]], col: Int): Array[Array[Float]] = {
    val min = data.map(_ (col)).min
    val max = data.map(_ (col)).max
    data.foreach { d =>
      d(col) = (d(col) - min) / (max - min)
    }
    data
  }
  def uploadKeys(keys: Array[String]): Unit = {
    val salt = flClient.psiStub.getSalt
    logger.debug("Client get Salt=" + salt)
    val hashedKeys = TestUtils.parallelToSHAHexString(keys, salt)
    hashedKeyPairs = hashedKeys.zip(keys).toMap
    // Hash(IDs, salt) into hashed IDs
    logger.debug("HashedIDs Size = " + hashedKeys.size)
    flClient.psiStub.uploadSet(hashedKeys.toList.asJava)

  }
  def getIntersectionKeys(): Array[String] = {
    require(null != hashedKeyPairs, "no hashed key pairs found, have you upload keys?")
    // TODO: just download
    var maxWait = 20
    var intersection: util.List[String] = null
    while (maxWait > 0) {
      intersection = flClient.psiStub.downloadIntersection()
      if (intersection == null || intersection.length == 0) {
        logger.info("Wait 1000ms")
        Thread.sleep(1000)
      } else {
        logger.info("Intersection successful. The id(s) in the intersection are: ")
        logger.info(intersection.mkString(", "))
        logger.info("Origin IDs are: ")
        logger.info(intersection.map(hashedKeyPairs(_)).mkString(", "))
        //break
        maxWait = 1
      }
      maxWait -= 1
    }
    intersection.asScala.toArray.map { k =>
      require(hashedKeyPairs.contains(k), "unknown intersection keys, please check psi server.")
      hashedKeyPairs(k)
    }
  }
}

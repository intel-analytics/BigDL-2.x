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

package com.intel.analytics.zoo.serving.baseline

import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.TestUtils
import com.intel.analytics.zoo.serving.serialization.JsonInputDeserializer
import com.intel.analytics.zoo.serving.utils.TimerSupportive
import org.apache.log4j.Logger
import com.codahale.metrics.MetricRegistry
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

object Operations extends TimerSupportive {
//  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    // read file from zoo/src/test/resources/serving to String
    // this is a prepared json format input of DIEN recommendation model
    val string = TestUtils.getStrFromResourceFile("dien_json_str_bs16.json")

    // decode json string input to activity
    val input = JsonInputDeserializer.deserialize(string)

    // create a InferenceModel and predict
    val model = new InferenceModel(1)
    model.doLoadTensorflow("/home/lyubing/models/dien", "frozenModel")
    val result = timing("predict")(predictRequestTimer) {
      model.doPredict(input)
    }

    // use 3 threads to predict each 100 times
    val model3 = new InferenceModel(3)
    model3.doLoadTensorflow("/home/lyubing/models/dien", "frozenModel")
    (0 until 3).indices.toParArray.foreach(threadIndex => {
      (0 until 100).foreach(i => {
        val input = JsonInputDeserializer.deserialize(string)
        timing(s"thread $threadIndex predict")(threadPredictRequestTimer) {
          val result = model3.doPredict(input)
        }
      })
    })
    logger.info("inference without sleep benchmark test ended.\n\n\n")
    // same operations above, but add some sleep after predict
    (0 until 3).indices.toParArray.foreach(threadIndex => {
      (0 until 100).foreach(i => {
        val input = JsonInputDeserializer.deserialize(string)
        timing(s"thread $threadIndex predict")(threadPredictRequestTimer) {
          val result = model3.doPredict(input)
          if (i % (threadIndex + 2) == 0) {
            // sleep for 100 ms
            Thread.sleep(100)
          }
        }
      })
    })
  }

  val metrics = new MetricRegistry
  val predictRequestTimer = metrics.timer("zoo.serving.request.predict")
  val threadPredictRequestTimer = metrics.timer("zoo.serving.request.predict.thread")

}

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
import com.intel.analytics.zoo.serving.serialization.JsonUtil
import org.apache.log4j.Logger
import com.codahale.metrics.MetricRegistry
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

object Operations extends TimerSupportive {
  // val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    // read file from zoo/src/test/resources/serving to String
    // this is a prepared json format input of DIEN recommendation model
    val string = TestUtils.getStrFromResourceFile("dien_json_str_bs16.json")

    // decode json string input to activity
     val input = JsonInputDeserializer.deserialize(string)

    (1 until 4).foreach(threadNumber => {
      // load model with concurrent number 1~4

      // set timer name
      val keyTotal = s"zoo.serving.request.predict.${threadNumber}_threads"
      // initialize timers
      val predictTotalRequestTimer = metrics.timer(keyTotal)

      val model = new InferenceModel(threadNumber)
      model.doLoadTensorflow("/home/lyubing/models/dien", "frozenModel")

      (0 until 10).foreach(iter =>  {
        // set timer name
        val key = s"zoo.serving.request.predict.${threadNumber}_threads.range_${iter}"
        // initialize timers
        val predictRequestTimer = metrics.timer(key)
        silent(s"inference with $threadNumber threads without sleep predict")(predictTotalRequestTimer,predictRequestTimer){
          // do a operation 0 to 10 times to mock preprocessing, the operation could be controlled around 1ms
          (0 until iter).foreach(i => {
            mockOperation()
          })
          // do predict
          (0 until threadNumber).indices.toParArray.foreach(threadIndex => {
            (0 until iter).foreach(i => {
              val result = model.doPredict(input)
            })
          })
          // do a operation 0 to 10 times to mock postprocessing
          (0 until iter).foreach(i => {
            mockOperation()
          })
          // sleep 0 to 10 ms
          Thread.sleep(iter)
        }
        val timer = metrics.getTimers().get(key)
        val servingMetrics = ServingTimerMetrics(key, timer)
        val jsonMetrics = JsonUtil.toJson(servingMetrics)
        logger.info(jsonMetrics)
        logger.info(s"inference with $threadNumber threads and range $iter benchmark test ended.\n")
      })
      val timer = metrics.getTimers().get(keyTotal)
      val servingMetrics = ServingTimerMetrics(keyTotal, timer)
      val jsonMetrics = JsonUtil.toJson(servingMetrics)
      logger.info(jsonMetrics)
      logger.info(s"inference with $threadNumber threads benchmark test ended.\n\n\n")
    })
  }

  val metrics = new MetricRegistry

}

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
import com.intel.analytics.zoo.serving.utils.Supportive
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

object Operations {
  def main(args: Array[String]): Unit = {
    // read file from zoo/src/test/resources/serving to String
    // this is a prepared json format input of DIEN recommendation model
    val string = TestUtils.getStrFromResourceFile("dien_json_str_bs16.json")

    // decode json string input to activity
    val input = JsonInputDeserializer.deserialize(string)

    // create a InferenceModel and predict
    val model = new InferenceModel(1)
    model.doLoadTensorflow("path/to/model", "frozenModel")
    val result = model.doPredict(input)

    // use 3 threads to do above operations
    val model3 = new InferenceModel(3)
    model3.doLoadTensorflow("path/to/model", "frozenModel")
    (0 until 3).indices.toParArray.foreach(threadIndex => {
      val input = JsonInputDeserializer.deserialize(string)
      val result = model3.doPredict(input)
    })
  }

}

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

package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, SerParams}
import org.scalatest.{FlatSpec, Matchers}

class InferenceSpec extends FlatSpec with Matchers {
  "TF String input" should "work" in {
//    val configPath = "/home/litchy/pro/analytics-zoo/config.yaml"
    val str = "abc|dff|aoa"
    val eleList = str.split("\\|")
//    val helper = new ClusterServingHelper(configPath)
//    helper.initArgs()
//    val param = new SerParams(helper)
//    val model = helper.loadInferenceModel()
//    val res = model.doPredict(t)
//    res
  }

}

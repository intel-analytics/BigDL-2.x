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
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, SerParams}
import org.scalatest.{FlatSpec, Matchers}

import sys.process._

class InferenceSpec extends FlatSpec with Matchers {
  "TF String input" should "work" in {
    ("wget -O /tmp/tf_string.tar http://10.239.45.10:8081" +
      "/repository/raw/analytics-zoo-data/tf_string.tar").!
    "tar -xvf /tmp/tf_string.tar -C /tmp/".!

    val model = new InferenceModel(1)
    val modelPath = "/tmp/tf_string"
//    val modelPath = "/home/litchy/models/tf_string"
    model.doLoadTensorflow(modelPath,
      "savedModel", null, null)

    ("rm -rf /tmp/tf_string*").!
    val t = Tensor[String](2)
    t.setValue(1, "123")
    t.setValue(2, "456")
    val res = model.doPredict(t)
    assert(res.toTensor[Float].valueAt(1) == 123)
    assert(res.toTensor[Float].valueAt(2) == 456)
  }

}

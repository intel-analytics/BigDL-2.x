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

package com.intel.analytics.zoo.models.textranker

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

class KNRMSpec extends ZooSpecHelper {

  "KNRM model" should "compute the correct output shape" in {
    val model = KNRM[Float](5, 10, 100, 20).buildModel()
    model.getOutputShape().toSingle().toArray should be (Array(-1, 1))
  }

  "KNRM forward and backward" should "work properly" in {
    val model = KNRM[Float](10, 20, 15, 10)
    val input = Tensor[Float](Array(2, 30)).rand(0.0, 0.95).apply1(x => (x*15).toInt)
    val output = model.forward(input)
    val gradInput = model.backward(input, output)
  }

  "KNRM with embedding weights batch=1 forward and backward" should "work properly" in {
    val weights = Tensor[Float](30, 100).rand()
    val model = KNRM[Float](15, 60, 30, 100, weights, 12, 0.2, 1e-4)
    val input = Tensor[Float](Array(1, 75)).rand(0.0, 1.0).apply1(x => (x*20).toInt)
    val output = model.forward(input)
    val gradInput = model.backward(input, output)
  }

}

class KNRMSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = KNRM[Float](10, 20, 15, 10)
    val input = Tensor[Float](Array(2, 30)).rand(0.0, 0.95).apply1(x => (x*15).toInt)
    ZooSpecHelper.testZooModelLoadSave(
      model.asInstanceOf[ZooModel[Tensor[Float], Tensor[Float], Float]],
      input, KNRM.loadModel[Float])
  }
}

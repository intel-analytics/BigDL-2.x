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

package com.intel.analytics.zoo.models.transformer

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

class TransformerSpec extends ZooSpecHelper {
  "Transformer model" should "be able to work" in {
    val model = Transformer[Float](vocab = 100, embeddingSize = 768)
    val input = Tensor[Float](Array(2, 2, 77, 2)).rand().resize(4, 77, 2)
    val gradOutput = Tensor[Float](4, 77, 768).rand()
    val output = model.forward(input)
    val gradInput = model.backward(input, output)
  }

  "Utils tril" should "be able to work" in {
    val data = Tensor.ones[Float](3, 3)
    Utils.tril(data)
    val expect = Array[Float](1, 0, 0, 1, 1, 0, 1, 1, 1)
    val res = data.storage().array()
    require(expect.deep == res.deep)

    val data2 = Tensor.ones[Float](4, 6)
    Utils.tril(data2)
    val expect2 = Array[Float](1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
      1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0)
    val res2 = data2.storage().array()
    require(expect2.deep == res2.deep)
  }
}

class TransformerSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val model = Transformer[Float]()
    val input = Tensor[Float](Array(1, 100, 50)).rand()
    ZooSpecHelper.testZooModelLoadSave(
      model.asInstanceOf[ZooModel[Tensor[Float], Tensor[Float], Float]],
      input, Transformer.loadModel[Float])
  }
}

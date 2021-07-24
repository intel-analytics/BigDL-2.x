/*
 * Copyright 2016 The BigDL Authors.
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
package com.intel.analytics.bigdl.common.utils.tf.loaders

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.common.utils.T
import com.intel.analytics.bigdl.common.utils.serializer.ModuleSerializationTest

import scala.util.Random


class TopKV2LoadTFSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val topkv2LoadTF = new TopKV2LoadTF[Float](false, "Float").
      setName("topkv2LoadTF")
    val input = T(Tensor[Float](3, 3).apply1(_ => Random.nextFloat()),
      Tensor.scalar[Int](2)
    )
    runSerializationTest(topkv2LoadTF, input)
  }
}

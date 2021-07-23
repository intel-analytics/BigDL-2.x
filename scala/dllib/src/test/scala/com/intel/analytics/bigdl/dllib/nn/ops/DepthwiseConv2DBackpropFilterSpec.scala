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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random


class DepthwiseConv2DBackpropFilterSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val depWiseConv2dBackProp = DepthwiseConv2DBackpropFilter[Float](1,
      1, 0, 0, DataFormat.NHWC).setName("depWiseConv2dBackProp")
    val input = T(Tensor[Float](4, 24, 24, 3).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(2, 2, 3, 1)),
      Tensor[Float](4, 23, 23, 3).apply1(_ => Random.nextFloat()))
    runSerializationTest(depWiseConv2dBackProp, input)
  }
}

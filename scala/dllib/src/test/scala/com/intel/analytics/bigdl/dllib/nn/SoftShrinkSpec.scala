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
package com.intel.analytics.bigdl.dllib.nn

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.common.utils.serializer.ModuleSerializationTest

import scala.util.Random

class SoftShrinkSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val softShrink = SoftShrink[Float]().setName("softShrink")
    val input = Tensor[Float](10, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(softShrink, input)
  }
}

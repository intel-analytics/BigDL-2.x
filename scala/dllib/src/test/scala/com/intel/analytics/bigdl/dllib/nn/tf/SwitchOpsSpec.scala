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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

class SwitchOpsSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val switchOps = new SwitchOps[Float]().setName("switchOps")
    val input =
      T(
        Tensor[Float](T(1.0f, 2.0f, 3.0f)),
        Tensor[Boolean](T(true))
      )
    runSerializationTest(switchOps, input)
  }
}

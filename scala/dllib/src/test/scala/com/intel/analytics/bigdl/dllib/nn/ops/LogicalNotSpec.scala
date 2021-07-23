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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class LogicalNotSpec extends FlatSpec with Matchers {
  "LogicalNot operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericBoolean
    val input = Tensor(T(true, false, true))

    val expectOutput = Tensor(T(false, true, false))

    val output = LogicalNot().forward(input)
    output should be(expectOutput)
  }
}

class LogicalNotSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val logicalNot = LogicalNot[Float].setName("logicalNot")
    val input = Tensor[Boolean](T(true, false))
    runSerializationTest(logicalNot, input, logicalNot
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}

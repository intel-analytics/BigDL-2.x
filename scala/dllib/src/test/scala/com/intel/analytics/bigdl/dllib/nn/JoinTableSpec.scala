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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class JoinTableSpec extends FlatSpec with Matchers {

  "Join Table " should "work for Int inputs" in {
    val input1 = Tensor[Int](T(1, 2))
    val input2 = Tensor[Int](T(3, 4))
    val layer = JoinTable[Float](1, -1)
    val expectedOutput = Tensor[Int](T(1, 2, 3, 4))
    val gradOuput = Tensor[Int](T(1, 2, 3, 4))
    val output = layer.forward(T(input1, input2))
    expectedOutput should be (output)
    val gradInput = layer.backward(T(input1, input2), gradOuput)
    gradInput[Tensor[Int]](1) should be (Tensor[Int](T(1, 2)))
    gradInput[Tensor[Int]](2) should be (Tensor[Int](T(3, 4)))
  }

}

class JoinTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val joinTable = JoinTable[Float](2, 2).setName("joinTable")
    val input1 = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](2, 2).apply1(_ => Random.nextFloat())
    val input = T()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(joinTable, input)
  }
}

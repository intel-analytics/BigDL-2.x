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
import org.scalatest.{FlatSpec, Matchers}

class SumSpec extends FlatSpec with Matchers {
  "Sum operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      Tensor(T(
        T(1f, 2f, 3f),
        T(2f, 2f, 4f),
        T(2f, 2f, 4f)
      ))

    val expectOutput = Tensor(T(5f, 6f, 11f))

    val output = Sum(axis = 1).forward(input)
    output should be(expectOutput)
  }
}

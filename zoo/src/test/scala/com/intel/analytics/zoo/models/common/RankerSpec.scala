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

package com.intel.analytics.zoo.models.common

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class RankerSpec extends FlatSpec with Matchers {

  "map" should "generate the correct answer" in {
    val output1 = Tensor[Float](Array(0.1f, 0.2f, 0.4f, 0.6f, 0.65f, 0.35f), Array(6, 1))
    val target1 = Tensor[Float](Array(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f), Array(6, 1))
    val res1 = Ranker.map[Float]().apply(output1, target1)
    require(res1 == 0.7) // The result is calculated from Python MatchZoo

    val output2 = Tensor[Float](Array(0.125f, 0.24f, 0.882f, 0.123f,
      0.754f, 0.123f, 0.187f, 0.298f, 0.125f, 0.779f), Array(10, 1))
    val target2 = Tensor[Float](Array(0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f), Array(10, 1))
    val res2 = Ranker.map[Float]().apply(output2, target2)
    require(res2.toFloat == 0.83333333f) // The result is calculated from Python MatchZoo
  }

  "ndcg" should "generate the correct answer" in {
    val output1 = Tensor[Float](Array(0.1f, 0.2f, 0.4f, 0.6f, 0.65f, 0.35f), Array(6, 1))
    val target1 = Tensor[Float](Array(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f), Array(6, 1))
    val res1_1 = Ranker.ndcg[Float](1).apply(output1, target1)
    require(res1_1 == 1.0) // The result is calculated from Python MatchZoo
    val res1_3 = Ranker.ndcg[Float](3).apply(output1, target1)
    require(res1_3.toFloat == 0.61314719f) // The result is calculated from Python MatchZoo

    val output2 = Tensor[Float](Array(0.125f, 0.24f, 0.882f, 0.123f,
      0.754f, 0.123f, 0.187f, 0.298f, 0.125f, 0.779f), Array(10, 1))
    val target2 = Tensor[Float](Array(1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f), Array(10, 1))
    val res2 = Ranker.ndcg[Float](3).apply(output2, target2)
    require(res2.toFloat == 0.70391809f) // The result is calculated from Python MatchZoo
  }

}

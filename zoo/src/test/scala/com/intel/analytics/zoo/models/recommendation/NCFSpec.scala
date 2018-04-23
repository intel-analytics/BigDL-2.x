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

package com.intel.analytics.zoo.models.recommendation

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

class NCFSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "NeuralCF without MF forward and backward" should "work properly" in {
    val userCount = 10
    val itemCount = 10
    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), false)
    val ran = new Random(42L)
    val data: Seq[Tensor[Float]] = (1 to 50).map(i => {
      val uid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val iid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val feature: Tensor[Float] = Tensor(T(T(uid, iid)))
      val label = Math.abs(ran.nextInt(4)).toFloat + 1
      feature
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

  "NeuralCF with MF forward and backward" should "work properly" in {
    val userCount = 10
    val itemCount = 10
    val model = NeuralCF[Float](userCount, itemCount, 5, 5, 5, Array(10, 5), true, 3)
    val ran = new Random(42L)
    val data: Seq[Tensor[Float]] = (1 to 50).map(i => {
      val uid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val iid = Math.abs(ran.nextInt(userCount - 1)).toFloat + 1
      val feature: Tensor[Float] = Tensor(T(T(uid, iid)))
      val label = Math.abs(ran.nextInt(4)).toFloat + 1
      feature
    })
    data.map { input =>
      val output = model.forward(input)
      val gradInput = model.backward(input, output)
    }
  }

}

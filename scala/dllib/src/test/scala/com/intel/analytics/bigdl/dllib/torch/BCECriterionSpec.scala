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
package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.BCECriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class BCECriterionSpec extends TorchSpec{

  "A BCECriterion " should "generate correct output and grad" in {
    torchCheck()
    val criterion = new BCECriterion[Double]()
    val input = Tensor[Double](3, 1).rand()
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 0
    target(Array(3)) = 1

    val start = System.nanoTime()
    val output1 = criterion.forward(input, target)
    val output2 = criterion.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "criterion = nn.BCECriterion()\n" +
      "output1 = criterion:forward(input, target)\n " +
      "output2 = criterion:backward(input, target)"


    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "target" -> target),
      Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Double]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)

    println("Test case : BCECriterion, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A BCECriterion with weights" should "generate correct output and grad" in {
    torchCheck()
    val weights = Tensor[Double](3).rand()
    val criterion = new BCECriterion[Double](weights)
    val input = Tensor[Double](3, 1).rand()
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 0
    target(Array(3)) = 1

    val start = System.nanoTime()
    val output1 = criterion.forward(input, target)
    val output2 = criterion.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "criterion = nn.BCECriterion(weights)\n" +
      "output1 = criterion:forward(input, target)\n " +
      "output2 = criterion:backward(input, target)"


    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "target" -> target, "weights" -> weights),
      Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Double]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)

    println("Test case : BCECriterion, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

  "A BCECriterion with sizeAverage" should "generate correct output and grad" in {
    torchCheck()
    val weights = Tensor[Double](3).rand()
    val criterion = new BCECriterion[Double](weights, true)
    val input = Tensor[Double](3, 1).rand()
    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 0
    target(Array(3)) = 1

    val start = System.nanoTime()
    val output1 = criterion.forward(input, target)
    val output2 = criterion.backward(input, target)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "criterion = nn.BCECriterion(weights, true)\n" +
      "output1 = criterion:forward(input, target)\n " +
      "output2 = criterion:backward(input, target)"


    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input, "target" -> target, "weights" -> weights),
      Array("output1", "output2"))
    val luaOutput1 = torchResult("output1").asInstanceOf[Double]
    val luaOutput2 = torchResult("output2").asInstanceOf[Tensor[Double]]

    luaOutput1 should be(output1)
    luaOutput2 should be(output2)

    println("Test case : BCECriterion, Torch : " + luaTime + " s, Scala : " +
      scalaTime / 1e9 + " s")
  }

}

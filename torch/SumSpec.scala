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

import com.intel.analytics.bigdl.nn.Sum
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator

@com.intel.analytics.bigdl.tags.Serial
class SumSpec extends TorchSpec {
    def randomn(): Double = RandomGenerator.RNG.normal(-10, 10)

  "An Sum()" should "generate correct output and grad" in {
    torchCheck()
    val layer = new Sum[Double]()
    val input = Tensor[Double](2, 2, 2)
    input.apply1(x => randomn())
    val gradOutput = Tensor[Double](1, 2, 2)
    gradOutput.apply1(x => randomn())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Sum()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Sum, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "An Sum(2)" should "generate correct output and grad" in {
    torchCheck()
    val layer = Sum[Double](2)
    val input = Tensor[Double](2, 2, 2)
    input.apply1(x => randomn())
    val gradOutput = Tensor[Double](1, 2, 2)
    gradOutput.apply1(x => randomn())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Sum(2)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Sum, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "An Sum(2,1,true)" should "generate correct output and grad" in {
    torchCheck()
    val layer = Sum[Double](2, 1, true)
    val input = Tensor[Double](2, 2, 2)
    input.apply1(x => randomn())
    val gradOutput = Tensor[Double](1, 2, 2)
    gradOutput.apply1(x => randomn())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Sum(2,1,true)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Sum, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }

  "An Sum(-1,1,true)" should "generate correct output and grad" in {
    torchCheck()
    val layer = Sum[Double](-1, 1, true)
    val input = Tensor[Double](2, 2, 2)
    input.apply1(x => randomn())
    val gradOutput = Tensor[Double](1, 2, 2)
    gradOutput.apply1(x => randomn())

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()
    val scalaTime = end - start

    val code = "module = nn.Sum(-1,1,true)\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Double]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Double]]

    output should be (luaOutput)
    gradInput should be (luaGradInput)

    println("Test case : Sum, Torch : " + luaTime + " s, Scala : " + scalaTime / 1e9 + " s")
  }
}

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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class SoftMaxSpec extends FlatSpec with Matchers {
  "SoftMax forward 1-D" should "work correctly" in {
    // we should test the cases which contain 1
    val tests = List(2, 1)

    for (x <- tests) {
      val sm = SoftMax()
      sm.evaluate()
      sm.setRuntime(new MklDnnRuntime)
      sm.initFwdPrimitives(Array(HeapData(Array(x), Memory.Format.x)), InferencePhase)

      val input = Tensor(x).rand()

      val output = sm.forward(input)

      val nnSm = nn.SoftMax()
      val nnOutput = nnSm.forward(input)

      Tools.dense(output) should be (nnOutput)
    }
  }

  "SoftMax forward 2-D" should "work correctly" in {
    val tests = List(
      (2, 3),
      (1, 3),
      (1, 1),
      (2, 1))

    for ((batchSize, channel) <- tests) {
      val sm = SoftMax()
      sm.setRuntime(new MklDnnRuntime)
      sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel), Memory.Format.nc)),
        InferencePhase)
      sm.evaluate()

      val input = Tensor(batchSize, channel).rand()

      val output = sm.forward(input)

      val nnSm = nn.SoftMax()
      val nnOutput = nnSm.forward(input)

      Tools.dense(output) shouldEqual nnOutput
    }
  }

  "SoftMax forward 4-D" should "work correctly" in {
    // we should test the cases which contain 1
    val tests = List(
      (2, 3, 4, 4),
      (1, 3, 4, 4),
      (1, 3, 1, 1),
      (1, 1, 1, 1),
      (1, 1, 3, 3),
      (2, 1, 3, 3),
      (2, 2, 1, 1))

    for ((batchSize, channel, height, width) <- tests) {
      val sm = SoftMax()
      sm.setRuntime(new MklDnnRuntime)
      sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
        Memory.Format.nchw)), InferencePhase)
      sm.evaluate()

      val input = Tensor(batchSize, channel, height, width).rand()

      val output = sm.forward(input)

      val nnSm = nn.SoftMax()
      val nnOutput = nnSm.forward(input)

      Tools.dense(output) should be (nnOutput)
    }
  }

  "SoftMax backward" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val sm = SoftMax()
    sm.setRuntime(new MklDnnRuntime)
    sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
      Memory.Format.nchw)), InferencePhase)

    val nnSm = nn.SoftMax()

    val input = Tensor(batchSize, channel, height, width).rand()
    val gradOutput = Tensor().resizeAs(input).rand(-10, 10)

    sm.forward(input)
    nnSm.forward(input)

    sm.backward(input, gradOutput)
    nnSm.backward(input, gradOutput)

    sm.output should be (nnSm.output)
    sm.gradInput should be (nnSm.gradInput)
  }

  "SoftMax multi times forward" should "work correctly" in {
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val sm = SoftMax()
    sm.setRuntime(new MklDnnRuntime)
    sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
      Memory.Format.nchw)), InferencePhase)
    sm.evaluate()

    val nnSm = nn.SoftMax()

    (0 until 5).foreach { _ =>
      val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
      sm.forward(input)
      nnSm.forward(input)

      Tools.dense(sm.output) should be (nnSm.output)
    }
  }
}

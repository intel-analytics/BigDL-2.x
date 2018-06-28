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
import com.intel.analytics.bigdl.nn.SpatialCrossMapLRN
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.BigDLSpecHelper
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

import scala.util.Random

class LRNSpec extends BigDLSpecHelper {
  "LRNDnn with format=nchw" should "work correctly" in {
    val batchSize = 2
    val input = Tensor[Float](batchSize, 7, 3, 3).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 7, 3, 3).apply1(e => Random.nextFloat())

    RNG.setSeed(100)
    val lrnDnn = LRN(5, 0.0001, 0.75, 1.0)
    RNG.setSeed(100)
    val lrnBLAS = SpatialCrossMapLRN[Float](5, 0.0001, 0.75, 1.0)

    val output2 = lrnBLAS.forward(input)
    val grad2 = lrnBLAS.updateGradInput(input, gradOutput)

    val seq = Sequential()
    seq.add(ReorderMemory(HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw),
      HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw)))
    seq.add(lrnDnn)
    seq.add(ReorderMemory(HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw),
      HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw)))
    seq.compile(Phase.TrainingPhase, Array(HeapData(Array(batchSize, 7, 3, 3), Memory.Format.nchw)))
    val output = seq.forward(input)
    output.asInstanceOf[Tensor[Float]] should be(output2)
    val grad1 = seq.backward(input, gradOutput)
    grad1.asInstanceOf[Tensor[Float]] should be(grad2)
  }

}

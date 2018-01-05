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

package com.intel.analytics.zoo.pipeline.common.nn

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class BboxPredSpec extends FlatSpec with Matchers {
  "BboxPred serializer" should "work properly" in {
    val module = BboxPred[Float](2, 3, nClass = 21)
    val input = Tensor[Float](3, 2).randn()
    val res1 = module.forward(input).clone()
    val tmpFile = java.io.File.createTempFile("module", ".bigdl")
    module.saveModule(tmpFile.getAbsolutePath, overWrite = true)
    val loaded = Module.loadModule[Float](tmpFile.getAbsolutePath)
    val res2 = loaded.forward(input)
    res1 should be(res2)
    if (tmpFile.exists()) {
      tmpFile.delete()
    }
  }

  "BboxPred evaluate serializer" should "work properly" in {
    val module = BboxPred[Float](2, 4, nClass = 1).evaluate()
    val input = Tensor[Float](4, 2).randn()
    val res1 = module.forward(input).clone()
    val tmpFile = java.io.File.createTempFile("module", ".bigdl")
    module.saveModule(tmpFile.getAbsolutePath, overWrite = true)
    val loaded = Module.loadModule[Float](tmpFile.getAbsolutePath).evaluate()
    val res2 = loaded.forward(input)
    res1 should be(res2)
    if (tmpFile.exists()) {
      tmpFile.delete()
    }
  }
}

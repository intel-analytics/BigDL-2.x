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
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

class AnchorTargetSpec extends FlatSpec with Matchers {
  "AnchorTarget serializer" should "work properly" in {
    val layer = AnchorTarget(Array[Float](1, 2, 3), Array[Float](0.1f, 0.2f, 0.3f))
    layer.setDebug(true)
    val rpnReg = Tensor(1, 36, 25, 38).randn()
    val gtBoxes = Tensor(7, 7).randn()
    val imInfo = Tensor(T(500, 600, 1, 1)).resize(1, 4)
    val input = T(rpnReg, gtBoxes, imInfo)
    val res1 = layer.forward(input)
    val tmpFile = java.io.File.createTempFile("module", ".bigdl")
    layer.saveModule(tmpFile.getAbsolutePath, true)
    val loadedAdd = Module.loadModule[Float](tmpFile.getAbsolutePath)
    val res2 = loadedAdd.forward(input).toTable
    res1 should be(res2)
    if (tmpFile.exists()) {
      tmpFile.delete()
    }
  }
}

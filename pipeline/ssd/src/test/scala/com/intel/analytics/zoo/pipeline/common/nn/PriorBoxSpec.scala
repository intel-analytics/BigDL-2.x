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

import com.intel.analytics.zoo.pipeline.ssd.model.ComponetParam
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.FlatSpec

class PriorBoxSpec extends FlatSpec {
  "Priorbox" should "work" in {
    val isClip = false
    val isFlip = true
    val variances = Array(0.1f, 0.1f, 0.2f, 0.2f)
    val param = ComponetParam(256, 4, minSizes = Array(460.8f),
      maxSizes = Array(537.6f), aspectRatios = Array(2), isFlip, isClip, variances, 512)
    val layer = PriorBox[Float](minSizes = param.minSizes, maxSizes = param.maxSizes,
      _aspectRatios = param.aspectRatios, isFlip = param.isFlip, isClip = param.isClip,
      variances = param.variances, step = param.step, offset = 0.5f, imgH = 512, imgW = 512)
    val input = Tensor[Float](8, 256, 1, 1)

    val out = layer.forward(input)

    val expectedStr = "0.0507812\n0.0507812\n0.949219\n0.949219\n0.0146376\n" +
      "0.0146376\n0.985362\n0.985362\n-0.135291\n0.182354\n1.13529\n0.817646\n" +
      "0.182354\n-0.135291\n0.817646\n1.13529\n0.1\n0.1\n0.2\n0.2\n0.1\n0.1\n0.2\n" +
      "0.2\n0.1\n0.1\n0.2\n0.2\n0.1\n0.1\n0.2\n0.2"

    val expected = Tensor(Storage(expectedStr.split("\n").map(_.toFloat))).resize(1, 2, 16)

    out.map(expected, (a, b) => {
      assert((a - b).abs < 1e-5);
      a
    })
  }
}

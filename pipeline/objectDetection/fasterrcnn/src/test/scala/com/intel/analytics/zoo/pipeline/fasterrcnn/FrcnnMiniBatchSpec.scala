/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.fasterrcnn

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.common.dataset.FrcnnMiniBatch


class FrcnnMiniBatchSpec extends FlatSpec with Matchers {
  "slice" should "work properly" in {
    val label = Tensor(Storage(Array(
      0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      1.0, 20.0, 0.0, 0.0603052, 0.368519, 0.848145, 1.0,
      1.0, 16.0, 0.0, 0.89412, 0.767627, 0.98189, 1.0,
      2.0, 16.0, 0.0, 0.645249, 0.577347, 0.731589, 0.808865,
      2.0, 15.0, 0.0, 0.614338, 0.646141, 0.850972, 0.83797,
      3.0, 8.0, 0.0, 0.241746, 0.322738, 0.447184, 0.478388,
      3.0, 8.0, 0.0, 0.318659, 0.336546, 0.661729, 0.675461,
      3.0, 8.0, 0.0, 0.56154, 0.300144, 0.699173, 0.708098,
      3.0, 8.0, 0.0, 0.220494, 0.327759, 0.327767, 0.396797,
      3.0, 8.0, 0.0, 0.194182, 0.317717, 0.279191, 0.389266,
      4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      5.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
      6.0, 10.0, 0.0, 0.67894, 0.471823, 0.929308, 0.632044,
      6.0, 10.0, 0.0, 0.381443, 0.572376, 0.892489, 0.691713,
      7.0, 9.0, 0.0, 0.0, 0.0620616, 0.667269, 1.0
    ).map(_.toFloat))).resize(15, 7)
    val input = T(T(Tensor(), Tensor(), label), T(Tensor(), Tensor(), label),
      T(Tensor(), Tensor(), label))

    val minibatch = FrcnnMiniBatch(input, label)
    val sub1 = minibatch.slice(1, 1)
    sub1.getTarget().toTensor[Float].size(1) should be (1)
    sub1.getTarget().toTensor[Float].valueAt(1, 1) should be (0)
    sub1.getInput().toTable[Tensor[Float]](3).size(1) should be (1)
    sub1.getInput().toTable[Tensor[Float]](3).valueAt(1, 1) should be (0)
  }
}

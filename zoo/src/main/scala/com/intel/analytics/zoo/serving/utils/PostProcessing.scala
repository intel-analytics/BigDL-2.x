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

package com.intel.analytics.zoo.serving.utils

import com.intel.analytics.bigdl.tensor.Tensor

object PostProcessing {
  def getInfofromTensor(topN: Int, result: Tensor[Float]): String = {

    val outputSize = if (result.size(1) > topN) {
      topN
    } else {
      result.size(1)
    }

    val output = TensorUtils.getTopN(outputSize, result)
    var value: String = "{"
    (0 until outputSize - 1).foreach( j => {
      val tmpValue = "\"" + output(j)._1 + "\":\"" +
        output(j)._2.toString + "\","
      value += tmpValue
    })
    value += "\"" + output(outputSize - 1)._1 + "\":\"" +
      output(outputSize - 1)._2.toString
    value += "\"}"
    value
  }

}

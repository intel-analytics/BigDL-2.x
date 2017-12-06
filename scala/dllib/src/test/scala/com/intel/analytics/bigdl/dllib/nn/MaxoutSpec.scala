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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.nn.Maxout
import com.intel.analytics.bigdl.tensor.Tensor

class MaxoutSpec extends KerasBaseSpec {
  "Maxout" should "generate corrent result when batchsize == 1" in {
    val inputSize = 2
    val outputSize = 4
    val maxoutNumber = 3
    val batchSize = 1

    val sigmoidCode =
      s"""
        |input_tensor = Input(shape=[${inputSize}])
        |input = np.random.uniform(0, 1, [${batchSize}, ${inputSize}])
        |output_tensor = MaxoutDense(output_dim=${outputSize}, input_dim=${inputSize},
        |nb_feature=${maxoutNumber})(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    val maxout = Maxout[Float](inputSize, outputSize, maxoutNumber)

    val wc = (data: Array[Tensor[Float]]) => {
      val out = new Array[Tensor[Float]](data.length)
      out(0) = Tensor(inputSize, maxoutNumber * outputSize)
      val weight = out.head.storage().array()
      var index = 0
      for (i <- 1 to maxoutNumber) {
        val sliceW = data(0).select(1, i).t.clone().storage().array()
        System.arraycopy(sliceW, 0, weight, index, sliceW.size)
        index += sliceW.size
      }

      if (data.length > 1) {
        out(1) = data(1)
      }
      out
    }
    checkOutputAndGrad(maxout, sigmoidCode, weightConverter = wc)
  }

  "Maxout" should "generate corrent result when batchsize != 1" in {
    val inputSize = 5
    val outputSize = 4
    val maxoutNumber = 3
    val batchSize = 4

    val sigmoidCode =
      s"""
        |#w1 = np.array([[[1.0, 2.0, 3.0, 4.0],
        |#               [5, 6, 7, 8.0]],
        |#              [[-1, -2, -3, -4],
        |#               [-5, -6, -7, -8]],
        |#              [[9, 10, 11, 12],
        |#               [-9, -10, -11, -12]]])
        |#b = np.array([[ 0.0,  0.0,  0.0,  0.0],
        |#       [ 0.0,  0.0,  0.0,  0.0],
        |#       [ 0.0,  0.0,  0.0,  0.0]])
        |# w = [w1, b]
        |
        |input_tensor = Input(shape=[${inputSize}])
        |input = np.random.uniform(0, 1, [${batchSize}, ${inputSize}])
        |#output_tensor=MaxoutDense(output_dim=4,input_dim=2,nb_feature=3,weights=w)(input_tensor)
        |output_tensor = MaxoutDense(output_dim=${outputSize}, input_dim=${inputSize},
        |nb_feature=${maxoutNumber})(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin

    val maxout = Maxout[Float](inputSize, outputSize, maxoutNumber)

    val wc = (data: Array[Tensor[Float]]) => {
      val out = new Array[Tensor[Float]](data.length)
      out(0) = Tensor(inputSize, maxoutNumber * outputSize)
      val weight = out.head.storage().array()
      var index = 0
      for (i <- 1 to maxoutNumber) {
        val sliceW = data(0).select(1, i).t.clone().storage().array()
        System.arraycopy(sliceW, 0, weight, index, sliceW.size)
        index += sliceW.size
      }

      if (data.length > 1) {
        out(1) = data(1)
      }
      out
    }
    checkOutputAndGrad(maxout, sigmoidCode, weightConverter = wc)
  }
}

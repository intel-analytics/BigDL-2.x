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

package com.intel.analytics.zoo.serving.engine

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.serving.PostProcessing
import com.intel.analytics.zoo.serving.utils.SerParams

object InferenceSupportive {
  def multiThreadInference(preProcessed: Iterator[(String, Tensor[Float])],
                           params: SerParams): Iterator[(String, String)] = {
    val postProcessed = preProcessed.grouped(params.coreNum).flatMap(pathByteBatch => {
      val thisBatchSize = pathByteBatch.size
      val t = if (params.chwFlag) {
        Tensor[Float](params.coreNum, params.C, params.H, params.W)
      } else {
        Tensor[Float](params.coreNum, params.H, params.W, params.C)
      }

      (0 until thisBatchSize).toParArray.foreach(i =>
        t.select(1, i + 1).copy(pathByteBatch(i)._2))

      val thisT = if (params.chwFlag) {
        t.resize(thisBatchSize, params.C, params.H, params.W)
      } else {
        t.resize(thisBatchSize, params.H, params.W, params.C)
      }
      val x = if (params.modelType == "openvino") {
        thisT.addSingletonDimension()
      } else {
        thisT
      }
      /**
       * addSingletonDimension method will modify the
       * original Tensor, thus if reuse of Tensor is needed,
       * have to squeeze it back.
       */
//      println(s"preparing to predict")
      val result = if (params.modelType == "openvino") {
        val res = params.model.doPredict(x).toTensor[Float].squeeze()
        t.squeeze(1)
        res
      } else {
        params.model.doPredict(x).toTensor[Float]
      }
//      println(s"predict end")
      (0 until thisBatchSize).toParArray.map(i => {
        val value = PostProcessing(result.select(1, i + 1), params.filter)
        (pathByteBatch(i)._1, value)
      })
    })
    postProcessed
  }

}

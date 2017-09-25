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

package com.intel.analytics.zoo.models

import com.intel.analytics.bigdl.nn.SoftMax
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.models.dataset.{ModelContext, PredictResult}
import com.intel.analytics.zoo.models.util.ImageNetLableReader
import org.apache.spark.rdd.RDD


trait Predictor {

  // def predictLocal(context : ModelContext, preprocessor: Preprocessor) : Array[PredictResult]

  def predictLocal(path : String, topNum : Int, preprocessor: Preprocessor = null)
  : Array[PredictResult]
 // def predict(context : ModelContext, preprocessor: Preprocessor) : RDD[Array[PredictResult]]

  def predictDistributed(paths : RDD[String], topNum : Int, preprocessor: Preprocessor = null):
    RDD[Array[PredictResult]]

  protected def topN(result : Tensor[Float], topN : Int) :
    Array[PredictResult] = {
    val preSoftMaxArray = result.storage().array()
    val preSoftMax = Tensor[Float](preSoftMaxArray, Array(preSoftMaxArray.length))
    val softMaxResult = SoftMax[Float]().forward(preSoftMax)
    val sortedResult = softMaxResult.storage().array().zipWithIndex.sortWith(_._1 > _._1).toList.toArray
    val res = new Array[PredictResult](topN)
    var index = 0
    while (index < topN) {
      val className = ImageNetLableReader.labelByIndex(sortedResult(index)._2 + 1)
      val credict = sortedResult(index)._1
      res(index) = PredictResult(className, credict)
      index += 1
    }
    res
  }
}




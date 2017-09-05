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

package com.intel.analytics.zoo.pipeline.ssd

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.rdd.RDD

//class Predictor(
//  model: Module[Float],
//  preProcessParam: PreProcessParam) {
//
//  val preProcessor =
//    RecordToByteRoiImage(false) ->
//      new BytesToMatImage() ->
//      Resize(preProcessParam.resolution, preProcessParam.resolution) ->
//      Normalize(preProcessParam.pixelMeanRGB) ->
//      new MatImageToFloats() ->
//      RoiImageToBatch(preProcessParam.batchSize, false, Some(preProcessParam.nPartition))
//
//  def predict(rdd: RDD[SSDByteRecord]): RDD[Tensor[Float]] = {
//    Predictor.predict(rdd, model, preProcessor)
//  }
//}
//
//object Predictor {
//  def predict(rdd: RDD[SSDByteRecord],
//    model: Module[Float],
//    preProcessor: Transformer[SSDByteRecord, SSDMiniBatch]): RDD[Tensor[Float]] = {
//    model.evaluate()
//    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
//    val broadcastTransformers = rdd.sparkContext.broadcast(preProcessor)
//    rdd.mapPartitions(dataIter => {
//      val localModel = broadcastModel.value()
//      val localTransformer = broadcastTransformers.value.cloneTransformer()
//      val miniBatch = localTransformer(dataIter)
//      miniBatch.flatMap(batch => {
//        val result = localModel.forward(batch.input).toTensor[Float]
//        BboxUtil.scaleBatchOutput(result, batch.imInfo)
//        result.split(1)
//      })
//    })
//  }
//}

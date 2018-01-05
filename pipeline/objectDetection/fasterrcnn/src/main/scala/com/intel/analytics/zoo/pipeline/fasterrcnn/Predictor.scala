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

package com.intel.analytics.zoo.pipeline.fasterrcnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.common.ModuleUtil
import com.intel.analytics.zoo.pipeline.common.dataset.{FrcnnMiniBatch, FrcnnToBatch}
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.{RecordToFeature, SSDByteRecord}
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.PreProcessParam
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{AspectScale, ChannelNormalize}
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, MatToFloats}
import org.apache.spark.rdd.RDD

class Predictor(
  model: Module[Float],
  preProcessParam: PreProcessParam) {

  val preProcessor = RecordToFeature(true) ->
    BytesToMat() ->
    AspectScale(preProcessParam.scales(0), preProcessParam.scaleMultipleOf) ->
    ChannelNormalize(preProcessParam.pixelMeanRGB._1,
      preProcessParam.pixelMeanRGB._2, preProcessParam.pixelMeanRGB._3) ->
    MatToFloats(100, 100) ->
    FrcnnToBatch(preProcessParam.batchSize, true, Some(preProcessParam.nPartition))

  ModuleUtil.shareMemory(model)

  def predict(rdd: RDD[SSDByteRecord]): RDD[Tensor[Float]] = {
    Predictor.predict(rdd, model, preProcessor)
  }
}

object Predictor {
  def predict(rdd: RDD[SSDByteRecord],
    model: Module[Float],
    preProcessor: Transformer[SSDByteRecord, FrcnnMiniBatch]
  ): RDD[Tensor[Float]] = {
    model.evaluate()
    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
    rdd.mapPartitions(preProcessor(_)).mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      dataIter.flatMap(batch => {
        val result = localModel.forward(batch.getSample()).toTensor[Float]
        if (result.dim() == 1) {
          Array(result)
        } else {
          result.split(1)
        }
      })
    })
  }
}

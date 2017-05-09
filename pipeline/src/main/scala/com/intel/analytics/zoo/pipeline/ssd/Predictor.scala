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
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage._
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.rdd.RDD

class Predictor(
    model: Module[Float],
    preProcessParam: PreProcessParam) {

  val preProcessor =
    RoiImageResizer(Array(preProcessParam.resolution), resizeRois = false, isEqualResize = true) ->
      RoiImageNormalizer(preProcessParam.pixelMeanRGB) ->
      RoiimageToBatch(preProcessParam.batchSize, false)

  def predict(rdd: RDD[RoiByteImage]): RDD[Array[Target]] = {
    Predictor.predict(rdd, model, preProcessor)
  }
}

object Predictor {
  def predict(rdd: RDD[RoiByteImage],
              model: Module[Float],
              preProcessor: Transformer[RoiByteImage, MiniBatch[Float]]): RDD[Array[Target]] = {
    model.evaluate()
    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
    rdd.mapPartitions(preProcessor(_)).mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      dataIter.map(batch => {
        val result = localModel.forward(batch.data).toTable
        scaleToOriginal(result(1).asInstanceOf[Array[Array[Target]]], batch.imInfo)
      }).flatten
    })
  }

  def scaleToOriginal(result: Array[Array[Target]], imInfo: Tensor[Float]): Array[Array[Target]] = {
    var i = 0
    val batch = result.length
    while (i < batch) {
      // Scale the bbox according to the original image size.
      val originalH = imInfo.valueAt(i + 1, 1) * imInfo.valueAt(i + 1, 3)
      val originalW = imInfo.valueAt(i + 1, 2) * imInfo.valueAt(i + 1, 4)
      val len = result(i).length
      var j = 0
      while (j < len) {
        if (result(i)(j) != null) BboxUtil.scaleBBox(result(i)(j).bboxes, originalH, originalW)
        j += 1
      }
      i += 1
    }
    result
  }
}

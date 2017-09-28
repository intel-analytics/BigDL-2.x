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
import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage._
import com.intel.analytics.zoo.pipeline.common.nn.DetectionOutput
import com.intel.analytics.zoo.pipeline.common.{BboxUtil, ModuleUtil, Predictor, Transform}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.ssd.model.PreProcessParam
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, ImageFeature, MatToFloats}
import com.intel.analytics.zoo.transform.vision.image.augmentation.Resize
import org.apache.spark.rdd.RDD

class SSDPredictor(
  model: Module[Float],
  preProcessParam: PreProcessParam, topK: Option[Int] = None) extends Serializable {

  ModuleUtil.shareMemory(model)

  if (topK.isDefined) {
    setTopK(model, topK)
  }

  private def setTopK(model: Module[Float], topK: Option[Int]): Unit = {
    if (topK.isEmpty) return
    val namedModules = Utils.getNamedModules(model)
    namedModules.values.foreach(layer => {
      if (layer.isInstanceOf[DetectionOutput[Float]]) {
        layer.asInstanceOf[DetectionOutput[Float]].setTopK(topK.get)
        return
      }
    })
  }

  private def postProcess(result: Tensor[Float], batch: SSDMiniBatch) =
    BboxUtil.scaleBatchOutput(result, batch.imInfo)

  def predict(rdd: RDD[SSDByteRecord]): RDD[Tensor[Float]] = {
    val preProcessor = RecordToFeature() ->
        BytesToMat() ->
        Resize(preProcessParam.resolution, preProcessParam.resolution) ->
        MatToFloats(validHeight = preProcessParam.resolution,
          validWidth = preProcessParam.resolution, meanRGB = Some(preProcessParam.pixelMeanRGB)) ->
        RoiImageToBatch(preProcessParam.batchSize, false, Some(preProcessParam.nPartition))

    val transformed = Transform(rdd, preProcessor)
    Predictor.predict(transformed, model, postProcess)
  }

  def predictWithFeature(rdd: RDD[ImageFeature]): RDD[ImageFeature] = {
    val preProcessor =
      BytesToMat() ->
      Resize(preProcessParam.resolution, preProcessParam.resolution) ->
        MatToFloats(validHeight = preProcessParam.resolution,
          validWidth = preProcessParam.resolution,
          meanRGB = Some(preProcessParam.pixelMeanRGB))
    val toBatch = RoiImageToBatch(preProcessParam.batchSize, false,
      Some(preProcessParam.nPartition))
    val transformed = Transform(rdd, preProcessor)
    Predictor.predict(transformed, model, "rois", toBatch, postProcess)
  }
}


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
import com.intel.analytics.bigdl.optim.ValidationMethod
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage._
import com.intel.analytics.zoo.pipeline.common.{BboxUtil, ModuleUtil}
import com.intel.analytics.zoo.pipeline.ssd.model.PreProcessParam
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, MatToFloats, RandomTransformer}
import com.intel.analytics.zoo.transform.vision.image.augmentation.Resize
import com.intel.analytics.zoo.transform.vision.label.roi.RoiNormalize
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD


class Validator(model: Module[Float],
  preProcessParam: PreProcessParam,
  evaluator: ValidationMethod[Float],
  useNormalized: Boolean = true
) {

  ModuleUtil.shareMemory(model)

  val normalizeRoi = if (useNormalized) RoiNormalize() else RandomTransformer(RoiNormalize(), 0)
  val preProcessor = RecordToFeature(true) ->
    BytesToMat() ->
    normalizeRoi ->
    Resize(preProcessParam.resolution, preProcessParam.resolution) ->
    MatToFloats(validHeight = preProcessParam.resolution,
      validWidth = preProcessParam.resolution, meanRGB = Some(preProcessParam.pixelMeanRGB)) ->
    RoiImageToBatch(preProcessParam.batchSize, true, Some(preProcessParam.nPartition))

  def test(rdd: RDD[SSDByteRecord]): Unit = {
    Validator.test(rdd, model, preProcessor, evaluator, useNormalized)
  }
}

object Validator {
  val logger = Logger.getLogger(this.getClass)

  def test(rdd: RDD[SSDByteRecord], model: Module[Float], preProcessor: Transformer[SSDByteRecord,
    SSDMiniBatch], evaluator: ValidationMethod[Float], useNormalized: Boolean = true): Unit = {
    model.evaluate()
    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
    val broadcastEvaluator = rdd.sparkContext.broadcast(evaluator)
    val broadcastTransformers = rdd.sparkContext.broadcast(preProcessor)
    val recordsNum = rdd.sparkContext.longAccumulator("record number")
    val start = System.nanoTime()
    val output = rdd.mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      val localEvaluator = broadcastEvaluator.value.clone()
      val localTransformer = broadcastTransformers.value.cloneTransformer()
      val miniBatch = localTransformer(dataIter)
      miniBatch.map(batch => {
        val result = localModel.forward(batch.input).toTensor[Float]
        if (!useNormalized) BboxUtil.scaleBatchOutput(result, batch.imInfo)
        recordsNum.add(batch.input.size(1))
        localEvaluator(result, batch.target)
      })
    }).reduce((left, right) => {
      left + right
    })
    logger.info(s"${evaluator} is ${output}")

    val totalTime = (System.nanoTime() - start) / 1e9
    logger.info(s"[Prediction] ${recordsNum.value} in $totalTime seconds. Throughput is ${
      recordsNum.value / totalTime
    } record / sec")
  }
}

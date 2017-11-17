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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.ValidationMethod
import com.intel.analytics.zoo.pipeline.common.ModuleUtil
import com.intel.analytics.zoo.pipeline.common.dataset.{FrcnnMiniBatch, FrcnnToBatch}
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.{RecordToFeature, SSDByteRecord}
import com.intel.analytics.zoo.pipeline.common.nn.FrcnnPostprocessor
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.{PostProcessParam, PreProcessParam}
import com.intel.analytics.zoo.transform.vision.image.augmentation.AspectScale
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, MatToFloats}
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD


class Validator(model: Module[Float],
  preProcessParam: PreProcessParam,
  postPrecessParam: PostProcessParam,
  evaluator: ValidationMethod[Float],
  shareMemory: Boolean = true) {
  val preProcessor = RecordToFeature(true) ->
    BytesToMat() ->
    AspectScale(preProcessParam.scales(0), preProcessParam.scaleMultipleOf) ->
    MatToFloats(100, 100, meanRGB = Some(preProcessParam.pixelMeanRGB)) ->
    FrcnnToBatch(preProcessParam.batchSize, true, Some(preProcessParam.nPartition))


  if (shareMemory) ModuleUtil.shareMemory(model)


  val postprocessor = Utils.getNamedModules(model)
    .find(x => x._2.isInstanceOf[FrcnnPostprocessor]).get._2.asInstanceOf[FrcnnPostprocessor]

  postprocessor.maxPerImage = postPrecessParam.maxPerImage
  postprocessor.nmsThresh = postPrecessParam.nmsThresh
  postprocessor.bboxVote = postPrecessParam.bboxVote
  postprocessor.thresh = postPrecessParam.thresh


  def test(rdd: RDD[SSDByteRecord]): Float = {
    Validator.test(rdd, model, preProcessor, evaluator)
  }
}

object Validator {
  val logger = Logger.getLogger(this.getClass)

  def test(rdd: RDD[SSDByteRecord], model: Module[Float],
    preProcessor: Transformer[SSDByteRecord, FrcnnMiniBatch],
    evaluator: ValidationMethod[Float]): Float = {
    model.evaluate()
    val broadcastModel = ModelBroadcast().broadcast(rdd.sparkContext, model)
    val broadcastEvaluator = rdd.sparkContext.broadcast(evaluator)
    val recordsNum = rdd.sparkContext.accumulator(0, "record number")
    val start = System.nanoTime()
    val output = rdd.mapPartitions(preProcessor(_)).mapPartitions(dataIter => {
      val localModel = broadcastModel.value()
      val localEvaluator = broadcastEvaluator.value
      dataIter.map(batch => {
        val result = localModel.forward(batch.getSample()).toTensor
        recordsNum += 1
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
    output.result()._1
  }
}

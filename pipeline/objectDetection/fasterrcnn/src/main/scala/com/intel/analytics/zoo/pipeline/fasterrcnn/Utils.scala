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

package com.intel.analytics.bigdl.pipeline.fasterrcnn

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.pipeline.common.IOUtils
import com.intel.analytics.zoo.pipeline.common.dataset.{FrcnnMiniBatch, FrcnnToBatch}
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.RecordToFeature
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.PreProcessParam
import com.intel.analytics.zoo.transform.vision.image.augmentation.{AspectScale, HFlip, RandomAspectScale}
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, MatToFloats, RandomTransformer}
import com.intel.analytics.zoo.transform.vision.label.roi._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


object Utils {
  def loadTrainSet(folder: String, sc: SparkContext, param: PreProcessParam, batchSize: Int)
  : DataSet[FrcnnMiniBatch] = {
    val trainRdd = IOUtils.loadSeqFiles(Engine.nodeNumber, folder, sc)._1
    DataSet.rdd(trainRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RandomAspectScale(param.scales, param.scaleMultipleOf) -> RoiResize() ->
      RandomTransformer(HFlip() -> RoiHFlip(false), 0.5) ->
      MatToFloats(validHeight = 600, validWidth = 600,
        meanRGB = Some(param.pixelMeanRGB)) ->
      FrcnnToBatch(batchSize, true)
  }

  def loadValSet(folder: String, sc: SparkContext, param: PreProcessParam, batchSize: Int)
  : DataSet[FrcnnMiniBatch] = {
    val valRdd = IOUtils.loadSeqFiles(Engine.nodeNumber, folder, sc)._1

    DataSet.rdd(valRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      AspectScale(param.scales(0), param.scaleMultipleOf) ->
      MatToFloats(100, 100, meanRGB = Some(param.pixelMeanRGB)) ->
      FrcnnToBatch(param.batchSize, true)
  }

  def loadValRdd(folder: String, sc: SparkContext,
    preProcessParam: PreProcessParam, batchSize: Int): RDD[FrcnnMiniBatch] = {
    val rdd = IOUtils.loadSeqFiles(preProcessParam.nPartition, folder, sc)._1
    val preProcessor = RecordToFeature(true) ->
      BytesToMat() ->
      AspectScale(preProcessParam.scales(0), preProcessParam.scaleMultipleOf) ->
      MatToFloats(100, 100, meanRGB = Some(preProcessParam.pixelMeanRGB)) ->
      FrcnnToBatch(preProcessParam.batchSize, true, Some(preProcessParam.nPartition))
    preProcessor(rdd)
  }
}
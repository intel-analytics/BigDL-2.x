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

package com.intel.analytics.zoo.models.image.objectdetection.ssd

import com.intel.analytics.zoo.feature.{DistributedFeatureSet, FeatureSet}
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.feature.image.roi.RoiRecordToFeature
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.Imdb
import org.apache.spark.SparkContext

object SSDDataSet {

  def loadSSDTrainSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int,
                      parNum: Option[Int])
  : DistributedFeatureSet[SSDMiniBatch] = {
    val trainRdd = Imdb.loadRoiSeqFiles(folder, sc, parNum)
    FeatureSet.rdd(trainRdd) -> RoiRecordToFeature(true) ->
      ImageBytesToMat() ->
      ImageRoiNormalize() ->
      ImageColorJitter() ->
      ImageRandomPreprocessing(ImageExpand() -> ImageRoiProject(), 0.5) ->
      ImageRandomSampler() ->
      ImageResize(resolution, resolution, -1) ->
      ImageRandomPreprocessing(ImageHFlip() -> ImageRoiHFlip(), 0.5) ->
      ImageChannelNormalize(123f, 117f, 104f) ->
      ImageMatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToSSDBatch(batchSize)
  }

  def loadSSDValSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int, parNum: Option[Int])
  : DistributedFeatureSet[SSDMiniBatch] = {
    val valRdd = Imdb.loadRoiSeqFiles(folder, sc, parNum)
    FeatureSet.rdd(valRdd) -> RoiRecordToFeature(true) ->
      ImageBytesToMat() ->
      ImageRoiNormalize() ->
      ImageResize(resolution, resolution) ->
      ImageChannelNormalize(123f, 117f, 104f) ->
      ImageMatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToSSDBatch(batchSize)
  }

}

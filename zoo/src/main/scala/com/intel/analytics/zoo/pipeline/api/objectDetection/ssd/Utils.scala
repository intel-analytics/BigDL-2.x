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

package com.intel.analytics.zoo.pipeline.api.objectDetection.ssd

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.label.roi._
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, MatToFloats}
import com.intel.analytics.zoo.pipeline.api.objectDetection.common.IOUtils
import com.intel.analytics.zoo.pipeline.api.objectDetection.common.dataset.roiimage._
import org.apache.spark.SparkContext


object Utils {

  def loadTrainSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int, parNum: Int)
  : DataSet[SSDMiniBatch] = {
    val trainRdd = IOUtils.loadSeqFiles(parNum, folder, sc)
    DataSet.rdd(trainRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      ColorJitter() ->
      RandomTransformer(Expand() -> RoiProject(), 0.5) ->
      RandomSampler() ->
      Resize(resolution, resolution, -1) ->
      RandomTransformer(HFlip() -> RoiHFlip(), 0.5) ->
      ChannelNormalize(123f, 117f, 104f) ->
      MatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToBatch(batchSize)
  }

  def loadValSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int, parNum: Int)
  : DataSet[SSDMiniBatch] = {
    val valRdd = IOUtils.loadSeqFiles(parNum, folder, sc)

    DataSet.rdd(valRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      Resize(resolution, resolution) ->
      ChannelNormalize(123f, 117f, 104f) ->
      MatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToBatch(batchSize)
  }
}


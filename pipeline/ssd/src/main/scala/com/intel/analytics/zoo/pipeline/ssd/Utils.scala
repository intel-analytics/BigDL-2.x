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

package com.intel.analytics.bigdl.pipeline.ssd

import java.io.File

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.zoo.pipeline.common.dataset.LocalByteRoiimageReader
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage._
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, MatToFloats, RandomTransformer}
import com.intel.analytics.zoo.transform.vision.image.augmentation._
import com.intel.analytics.zoo.transform.vision.label.roi.{RandomSampler, RoiExpand, RoiHFlip, RoiNormalize}
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


object IOUtils {
  def loadSeqFiles(nPartition: Int, seqFloder: String, sc: SparkContext)
  : (RDD[SSDByteRecord], RDD[String]) = {
    val data = sc.sequenceFile(seqFloder, classOf[Text], classOf[Text],
      nPartition).map(x => SSDByteRecord(x._2.copyBytes(), x._1.toString))
    val paths = data.map(x => x.path)
    (data, paths)
  }

  def loadLocalFolder(nPartition: Int, folder: String, sc: SparkContext)
  : (RDD[SSDByteRecord], RDD[String]) = {
    val roiDataset = localImagePaths(folder).map(RoiImagePath(_))
    val imgReader = LocalByteRoiimageReader()
    val data = sc.parallelize(roiDataset.map(roidb => imgReader.transform(roidb)),
      nPartition)
    (data, data.map(_.path))
  }

  def localImagePaths(folder: String): Array[String] = {
    new File(folder).listFiles().map(_.getAbsolutePath)
  }

  def loadTrainSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int)
  : DataSet[SSDMiniBatch] = {
    val trainRdd = IOUtils.loadSeqFiles(Engine.nodeNumber, folder, sc)._1
    DataSet.rdd(trainRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      ColorJitter() ->
      RandomTransformer(Expand() -> RoiExpand(), 0.5) ->
      RandomSampler() ->
      Resize(resolution, resolution, -1) ->
      RandomTransformer(HFlip() -> RoiHFlip(), 0.5) ->
      MatToFloats(validHeight = resolution, validWidth = resolution,
        meanRGB = Some(123f, 117f, 104f)) ->
      RoiImageToBatch(batchSize)
  }

  def loadValSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int)
  : DataSet[SSDMiniBatch] = {
    val valRdd = IOUtils.loadSeqFiles(Engine.nodeNumber, folder, sc)._1

    DataSet.rdd(valRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      Resize(resolution, resolution) ->
      MatToFloats(validHeight = resolution, validWidth = resolution,
        meanRGB = Some(123f, 117f, 104f)) ->
      RoiImageToBatch(batchSize)
  }
}


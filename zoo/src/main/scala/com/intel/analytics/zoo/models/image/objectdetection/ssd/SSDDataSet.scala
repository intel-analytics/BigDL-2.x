package com.intel.analytics.zoo.models.image.objectdetection.ssd

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet

import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.feature.image.roi.RoiRecordToFeature
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.Imdb
import org.apache.spark.SparkContext

object SSDDataSet {

  def loadSSDTrainSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int,
                      parNum: Option[Int])
  : DataSet[SSDMiniBatch] = {
    val trainRdd = Imdb.loadRoiSeqFiles(folder, sc, parNum)
    DataSet.rdd(trainRdd) -> RoiRecordToFeature(true) ->
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
  : DataSet[SSDMiniBatch] = {
    val valRdd = Imdb.loadRoiSeqFiles(folder, sc, parNum)
    DataSet.rdd(valRdd) -> RoiRecordToFeature(true) ->
      ImageBytesToMat() ->
      ImageRoiNormalize() ->
      ImageResize(resolution, resolution) ->
      ImageChannelNormalize(123f, 117f, 104f) ->
      ImageMatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToSSDBatch(batchSize)
  }

}

package com.intel.analytics.zoo.models.image.objectdetection.fasterrcnn

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.zoo.models.image.objectdetection.common.IOUtils
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.label.roi._
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.{FrcnnMiniBatch,
        FrcnnToBatch}
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.roiimage.RecordToFeature
import org.apache.spark.SparkContext

object Utils {
  def loadTrainSet(folder: String, sc: SparkContext, param: PreProcessParam, batchSize: Int, parNum: Int)
  : DataSet[FrcnnMiniBatch] = {
    val trainRdd = IOUtils.loadSeqFiles(parNum, folder, sc)
    DataSet.rdd(trainRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      RandomAspectScale(param.scales, param.scaleMultipleOf) -> RoiResize() ->
      RandomTransformer(HFlip() -> RoiHFlip(false), 0.5) ->
      ChannelNormalize(param.pixelMeanRGB._1, param.pixelMeanRGB._2, param.pixelMeanRGB._3) ->
      MatToFloats(validHeight = 600, validWidth = 600) ->
      FrcnnToBatch(batchSize, true)
  }

  def loadValSet(folder: String, sc: SparkContext, param: PreProcessParam, batchSize: Int, parNum: Int)
  : DataSet[FrcnnMiniBatch] = {
    val valRdd = IOUtils.loadSeqFiles(parNum, folder, sc)

    DataSet.rdd(valRdd) -> RecordToFeature(true) ->
      BytesToMat() ->
      AspectScale(param.scales(0), param.scaleMultipleOf) ->
      ChannelNormalize(param.pixelMeanRGB._1, param.pixelMeanRGB._2, param.pixelMeanRGB._3) ->
      MatToFloats(100, 100) ->
      FrcnnToBatch(param.batchSize, true)
  }
}

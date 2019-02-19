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

package com.intel.analytics.zoo.models.image.objectdetection.common

import java.io.File

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import com.intel.analytics.bigdl.transform.vision.image.label.roi._
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFeature, ImageFrame, MatToFloats}
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.{FrcnnMiniBatch, FrcnnToBatch}
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.roiimage.{ByteRecord, RoiImageToBatch, RoiRecordToFeature, SSDMiniBatch}
import com.intel.analytics.zoo.models.image.objectdetection.fasterrcnn
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag


object IOUtils {
  def loadRoiSeqFiles(seqFloder: String, sc: SparkContext, nPartition: Option[Int] = None): RDD[ByteRecord] = {
    val raw = if(nPartition.isDefined) {
      sc.sequenceFile(seqFloder, classOf[Text], classOf[Text], nPartition.get)
    } else {
      sc.sequenceFile(seqFloder, classOf[Text], classOf[Text])
    }
    raw.map(x => ByteRecord(x._2.copyBytes(), x._1.toString))
  }

  def roiSeqFilesToImageFrame(url: String, sc: SparkContext, partitionNum: Option[Int] = None): ImageFrame = {
    val rdd = loadRoiSeqFiles(url, sc, partitionNum)
    val featureRDD = RoiRecordToFeature(true)(rdd)
    ImageFrame.rdd(featureRDD)
  }

  def localImagePaths(folder: String): LocalImageSet = {
    val arr = new File(folder).listFiles().map(x => {
      val imf = ImageFeature()
      imf(ImageFeature.uri) = x.getAbsolutePath
      imf
    })
    ImageSet.array(arr)
  }

  def loadSSDTrainSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int,
                      parNum: Int)
  : DataSet[SSDMiniBatch] = {
//    val trainRdd = loadSeqFiles(parNum, folder, sc)
    val imageFrame = roiSeqFilesToImageFrame(folder, sc)
    DataSet.imageFrame(imageFrame) ->
      ImageBytesToMat() ->
      ImageRoiNormalize() ->
      ImageColorJitter() ->
      ImageRandomPreprocessing(ImageExpand() -> ImageRoiProject(), 0.5) ->
      ImageRandomSampler() ->
      ImageResize(resolution, resolution, -1) ->
      ImageRandomPreprocessing(ImageHFlip() -> ImageRoiHFlip(), 0.5) ->
      ImageChannelNormalize(123f, 117f, 104f) ->
      ImageMatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToBatch(batchSize)
  }

  def loadSSDValSet(folder: String, sc: SparkContext, resolution: Int, batchSize: Int, parNum: Int)
  : DataSet[SSDMiniBatch] = {
//    val valRdd = loadSeqFiles(parNum, folder, sc)
    val imageFrame = roiSeqFilesToImageFrame(folder, sc)
    DataSet.imageFrame(imageFrame) ->
      ImageBytesToMat() ->
      ImageRoiNormalize() ->
      ImageResize(resolution, resolution) ->
      ImageChannelNormalize(123f, 117f, 104f) ->
      ImageMatToFloats(validHeight = resolution, validWidth = resolution) ->
      RoiImageToBatch(batchSize)
  }

  def loadFasterrcnnTrainSet(folder: String, sc: SparkContext, param: fasterrcnn.PreProcessParam,
                             batchSize: Int, parNum: Int)
  : DataSet[FrcnnMiniBatch] = {
    val trainRdd = loadRoiSeqFiles(folder, sc, Some(parNum))
//    val imageFrame = roiSeqFilesToImageFrame(folder, sc)
    DataSet.rdd(trainRdd) -> RoiRecordToFeature(true) ->
      ImageBytesToMat() ->
      ImageRandomAspectScale(param.scales, param.scaleMultipleOf) -> RoiResize() ->
      ImageRandomPreprocessing(ImageHFlip() -> ImageRoiHFlip(false), 0.5) ->
      ImageChannelNormalize(param.pixelMeanRGB._1, param.pixelMeanRGB._2, param.pixelMeanRGB._3) ->
      ImageMatToFloats(validHeight = 600, validWidth = 600) ->
      FrcnnToBatch(batchSize, true)
  }

  def loadFasterrcnnValSet(folder: String, sc: SparkContext, param: fasterrcnn.PreProcessParam,
                           batchSize: Int, parNum: Int)
  : DataSet[FrcnnMiniBatch] = {
    val valRdd = loadRoiSeqFiles(folder, sc, Some(parNum))
    DataSet.rdd(valRdd) -> RoiRecordToFeature(true) ->
      ImageBytesToMat() ->
      ImageAspectScale(param.scales(0), param.scaleMultipleOf) ->
      ImageChannelNormalize(param.pixelMeanRGB._1, param.pixelMeanRGB._2, param.pixelMeanRGB._3) ->
      ImageMatToFloats(100, 100) ->
      FrcnnToBatch(param.batchSize, true)
  }
}

object OBUtils {
  def addConvRelu[@specialized(Float, Double) T: ClassTag](prevNodes: ModuleNode[T],
    p: (Int, Int, Int, Int, Int), name: String, prefix: String = "conv", nGroup: Int = 1,
    propogateBack: Boolean = true)(implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    val conv = SpatialConvolution[T](p._1, p._2, p._3, p._3, p._4, p._4,
      p._5, p._5, nGroup = nGroup, propagateBack = propogateBack)
      .setInitMethod(weightInitMethod = Xavier, biasInitMethod = Zeros)
      .setName(s"$prefix$name").inputs(prevNodes)
    ReLU[T](true).setName(s"relu$name").inputs(conv)
  }

  def stopGradient[@specialized(Float, Double) T: ClassTag](model: Graph[T]): Unit = {
    val priorboxNames = model.modules
      .filter(_.getClass.getName.toLowerCase().endsWith("priorbox"))
      .map(_.getName()).toArray
    model.stopGradient(priorboxNames)
  }
}

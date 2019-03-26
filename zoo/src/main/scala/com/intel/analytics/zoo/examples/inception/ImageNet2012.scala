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

package com.intel.analytics.zoo.examples.inception

import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.CropCenter
import com.intel.analytics.bigdl.dataset.image.{BGRImgCropper, BGRImgNormalizer, BytesToBGRImg, MTLabeledBGRImgToBatch, HFlip => DatasetHFlip}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.feature.{DistributedFeatureSet, FeatureSet}
import com.intel.analytics.zoo.feature.pmem.{DRAM, MemoryType, PMEM}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.hadoop.io.Text
import org.apache.log4j.Logger
import org.apache.spark.SparkContext

object ImageNet2012 {
  val logger = Logger.getLogger(this.getClass)

  /**
   * Extract hadoop sequence files from an HDFS path
   *
   * @param url
   * @param sc
   * @param classNum
   * @return
   */
  private[inception] def readFromSeqFiles(
        url: String, sc: SparkContext, classNum: Int) = {
    val nodeNumber = EngineRef.getNodeNumber()
    val coreNumber = EngineRef.getCoreNumber()
    val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text],
      nodeNumber * coreNumber).map(image => {
      ByteRecord(image._2.copyBytes(), readLabel(image._1).toFloat)
    }).filter(_.label <= classNum)
    rawData
  }

  /**
   * get label from text of sequence file,
   *
   * @param data text of sequence file, this text can split into parts by "\n"
   * @return
   */
  private def readLabel(data: Text): String = {
    val dataArr = data.toString.split("\n")
    if (dataArr.length == 1) {
      dataArr(0)
    } else {
      dataArr(1)
    }
  }

  def apply(
    path : String,
    sc: SparkContext,
    imageSize : Int,
    batchSize : Int,
    nodeNumber: Int,
    coresPerNode: Int,
    classNumber: Int,
    memoryType: MemoryType = DRAM,
    opencvPreprocessing: Boolean = false
  )
  : FeatureSet[MiniBatch[Float]] = {
    if (opencvPreprocessing) {
      logger.info("Using opencv preprocessing for training set")
      opencv(path, sc, imageSize, batchSize,
        nodeNumber, coresPerNode, classNumber, memoryType)
    } else {
      val rawData = readFromSeqFiles(path, sc, classNumber)
        .setName("ImageNet2012 Training Set")
      val featureSet = FeatureSet.rdd(rawData, memoryType = memoryType)
      featureSet.transform(
        MTLabeledBGRImgToBatch[ByteRecord](
          width = imageSize,
          height = imageSize,
          batchSize = batchSize,
          transformer = (BytesToBGRImg()
            -> BGRImgCropper(imageSize, imageSize)
            -> DatasetHFlip(0.5)
            -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225))
        ))
    }
  }

  private[inception] def byteRecordToImageFeature(record: ByteRecord): ImageFeature = {
    val rawBytes = record.data
    val label = Tensor[Float](T(record.label))
    val imgBuffer = ByteBuffer.wrap(rawBytes)
    val width = imgBuffer.getInt
    val height = imgBuffer.getInt
    val bytes = new Array[Byte](3 * width * height)
    System.arraycopy(imgBuffer.array(), 8, bytes, 0, bytes.length)
    val imf = ImageFeature(bytes, label)
    imf(ImageFeature.originalSize) = (height, width, 3)
    imf
  }

  def opencv(
        path : String,
        sc: SparkContext,
        imageSize : Int,
        batchSize : Int,
        nodeNumber: Int,
        coresPerNode: Int,
        classNumber: Int,
        memoryType: MemoryType = DRAM): FeatureSet[MiniBatch[Float]] = {
    val rawData = readFromSeqFiles(path, sc, classNumber)
      .map(byteRecordToImageFeature(_))
      .setName("ImageNet2012 Training Set")
    val featureSet = FeatureSet.rdd(rawData, memoryType = memoryType)
    val transformer = ImagePixelBytesToMat() ->
      ImageRandomCrop(imageSize, imageSize) ->
      ImageChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      ImageMatToTensor[Float](true) ->
      ImageSetToSample[Float](inputKeys = Array(ImageFeature.imageTensor),
        targetKeys = Array(ImageFeature.label)) ->
      ImageFeatureToSample[Float]() ->
      SampleToMiniBatch[Float](batchSize)
    featureSet.transform(transformer)
  }
}


object ImageNet2012Val {
  val logger = Logger.getLogger(this.getClass)

  def apply(
    path : String,
    sc: SparkContext,
    imageSize : Int,
    batchSize : Int,
    nodeNumber: Int,
    coresPerNode: Int,
    classNumber: Int,
    memoryType: MemoryType = DRAM,
    opencvPreprocessing: Boolean = false
  ): FeatureSet[MiniBatch[Float]] = {
    if (opencvPreprocessing) {
      logger.info("Using opencv preprocessing for validation set")
      opencv(path, sc, imageSize, batchSize,
        nodeNumber, coresPerNode, classNumber, memoryType)
    } else {
      val rawData = ImageNet2012.readFromSeqFiles(path, sc, classNumber)
        .setName("ImageNet2012 Validation Set")
      val featureSet = FeatureSet.rdd(rawData, memoryType = memoryType)
      featureSet.transform(
        MTLabeledBGRImgToBatch[ByteRecord](
          width = imageSize,
          height = imageSize,
          batchSize = batchSize,
          transformer = (BytesToBGRImg()
            -> BGRImgCropper(imageSize, imageSize, CropCenter)
            -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225))
        ))
    }
  }

  def opencv(
        path : String,
        sc: SparkContext,
        imageSize : Int,
        batchSize : Int,
        nodeNumber: Int,
        coresPerNode: Int,
        classNumber: Int,
        memoryType: MemoryType = DRAM): FeatureSet[MiniBatch[Float]] = {
    val rawData = ImageNet2012.readFromSeqFiles(path, sc, classNumber)
      .map(ImageNet2012.byteRecordToImageFeature(_))
      .setName("ImageNet2012 Validation Set")
    val featureSet = FeatureSet.rdd(rawData, memoryType = memoryType)
    val transformer = ImagePixelBytesToMat() ->
      ImageCenterCrop(imageSize, imageSize) ->
      ImageChannelNormalize(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f) ->
      ImageMatToTensor[Float](true) ->
      ImageSetToSample[Float](inputKeys = Array(ImageFeature.imageTensor),
        targetKeys = Array(ImageFeature.label)) ->
      ImageFeatureToSample[Float]() -> SampleToMiniBatch[Float](batchSize)
    featureSet.transform(transformer)
  }

}

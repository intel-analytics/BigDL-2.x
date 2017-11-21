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

package com.intel.analytics.zoo.pipeline.common.dataset.roiimage

import com.intel.analytics.bigdl.dataset.{Utils}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.zoo.transform.vision.image.{ImageFeature, ImageFeatureToBatch}
import com.intel.analytics.zoo.transform.vision.label.roi.RoiLabel

import scala.collection.Iterator
import scala.collection.mutable.ArrayBuffer

object RoiImageToBatch {
  def apply(batchSize: Int, convertLabel: Boolean = true,
    partitionNum: Option[Int] = None, keepImageFeature: Boolean = true,
    inputKey: String = ImageFeature.floats): RoiImageToBatch
  = new RoiImageToBatch(batchSize, convertLabel, partitionNum, keepImageFeature, inputKey)
}

/**
 * Convert a batch of labeled BGR images into a Mini-batch.
 *
 * Notice: The totalBatch means a total batch size. In distributed environment, the batch should be
 * divided by total core number
 * @param totalBatch
 */
class RoiImageToBatch(totalBatch: Int,
  convertLabel: Boolean = true,
  partitionNum: Option[Int] = None, val keepImageFeature: Boolean = true,
  inputKey: String = ImageFeature.floats)
  extends ImageFeatureToBatch[SSDMiniBatch] {

  private val batchPerPartition = Utils.getBatchSize(totalBatch, partitionNum)

  override def apply(prev: Iterator[ImageFeature]): Iterator[SSDMiniBatch] = {
    val batchSizePerPartition = batchPerPartition
    new Iterator[SSDMiniBatch] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private val imInfoTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: ArrayBuffer[Float] = null
      private var imInfoData: Array[Float] = null
      private var maps: ArrayBuffer[ImageFeature] = null
      private var width = 0
      private var height = 0
      private val batchSize = batchSizePerPartition

      override def hasNext: Boolean = prev.hasNext

      override def next(): SSDMiniBatch = {
        if (prev.hasNext) {
          var i = 0
          if (labelData != null) labelData.clear()
          if (maps != null) maps.clear()
          while (i < batchSize && prev.hasNext) {
            val feature = prev.next()
            height = feature.getHeight()
            width = feature.getWidth()
            if (featureData == null) {
              featureData = new Array[Float](batchSize * 3 * height * width)
              imInfoData = new Array[Float](batchSize * 4)
              maps = new ArrayBuffer[ImageFeature]()
              if (convertLabel) {
                labelData = new ArrayBuffer[Float]()
              }
            }
            feature.copyTo(featureData, i * width * height * 3, inputKey, false)
            val imInfo = feature.getImInfo()
            imInfoData(i * 4) = imInfo.valueAt(1)
            imInfoData(i * 4 + 1) = imInfo.valueAt(2)
            imInfoData(i * 4 + 2) = imInfo.valueAt(3)
            imInfoData(i * 4 + 3) = imInfo.valueAt(4)
            if (convertLabel) {
              require(feature.hasLabel(), "if convert label, there should be label")
              val target = feature.getLabel[RoiLabel]
              if (target.classes.nElement() > 0) {
                var r = 0
                while (r < target.classes.size(2)) {
                  labelData.append(i)
                  labelData.append(target.classes.valueAt(1, r + 1))
                  // difficult
                  labelData.append(target.classes.valueAt(2, r + 1))
                  labelData.append(target.bboxes.valueAt(r + 1, 1))
                  labelData.append(target.bboxes.valueAt(r + 1, 2))
                  labelData.append(target.bboxes.valueAt(r + 1, 3))
                  labelData.append(target.bboxes.valueAt(r + 1, 4))
                  r += 1
                }
              } else {
                labelData.append(i)
                labelData.append(-1)
                // difficult
                labelData.append(-1)
                labelData.append(-1)
                labelData.append(-1)
                labelData.append(-1)
                labelData.append(-1)
              }
            }
            maps.append(feature)
            i += 1
          }

          if (featureTensor.nElement() != i * 3 * height * width) {
            featureTensor.set(Storage[Float](featureData),
              storageOffset = 1, sizes = Array(i, 3, height, width))
            imInfoTensor.set(Storage[Float](imInfoData), storageOffset = 1, sizes = Array(i, 4))
          }
          val batch = if (convertLabel) {
            labelTensor.set(Storage[Float](labelData.toArray),
              storageOffset = 1, sizes = Array(labelData.length / 7, 7))
            SSDMiniBatch(featureTensor, labelTensor, imInfoTensor)
          } else {
            SSDMiniBatch(featureTensor, null, imInfoTensor)
          }
          if (keepImageFeature) {
            batch.imageFeatures = maps
          }
          batch
        } else {
          null
        }
      }
    }
  }

  // todo: override it
  override def inputToBatch(imageFeatures: ArrayBuffer[ImageFeature]): Activity = null
}

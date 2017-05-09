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

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.Iterator
import scala.collection.mutable.ArrayBuffer

object RoiimageToBatch {
  def apply(batchSize: Int, convertLabel: Boolean = true): RoiimageToBatch
  = new RoiimageToBatch(batchSize, convertLabel)
}

/**
 * Convert a batch of labeled BGR images into a Mini-batch.
 *
 * Notice: The totalBatch means a total batch size. In distributed environment, the batch should be
 * divided by total core number
 * @param batchSize
 */
class RoiimageToBatch(batchSize: Int, convertLabel: Boolean = true)
  extends Transformer[RoiImage, MiniBatch[Float]] {

  override def apply(prev: Iterator[RoiImage]): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private val imInfoTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: ArrayBuffer[Float] = null
      private var imInfoData: Array[Float] = null
      private var width = 0
      private var height = 0

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[Float] = {
        if (prev.hasNext) {
          var i = 0
          if (labelData != null) labelData.clear()
          while (i < batchSize && prev.hasNext) {
            val img = prev.next()
            if (featureData == null) {
              height = img.height()
              width = img.width()
              featureData = new Array[Float](batchSize * 3 * height * width)
              imInfoData = new Array[Float](batchSize * img.imInfo.size(1))
              if (convertLabel) {
                labelData = new ArrayBuffer[Float]()
              }
            }
            img.copyTo(featureData, i * img.width() * img.height() * 3, false)
            imInfoData(i * 4) = img.imInfo.valueAt(1)
            imInfoData(i * 4 + 1) = img.imInfo.valueAt(2)
            imInfoData(i * 4 + 2) = img.imInfo.valueAt(3)
            imInfoData(i * 4 + 3) = img.imInfo.valueAt(4)
            if (convertLabel) {
              var r = 0
              while (img.target != null && img.target.classes.nElement() > 0
                && r < img.target.classes.size(2)) {
                labelData.append(i)
                labelData.append(img.target.classes.valueAt(1, r + 1))
                // difficult
                labelData.append(img.target.classes.valueAt(2, r + 1))
                labelData.append(img.target.bboxes.valueAt(r + 1, 1))
                labelData.append(img.target.bboxes.valueAt(r + 1, 2))
                labelData.append(img.target.bboxes.valueAt(r + 1, 3))
                labelData.append(img.target.bboxes.valueAt(r + 1, 4))
                r += 1
              }
            }
            i += 1
          }

          if (featureTensor.nElement() != i * 3 * height * width) {
            featureTensor.set(Storage[Float](featureData),
              storageOffset = 1, sizes = Array(i, 3, height, width))
            imInfoTensor.set(Storage[Float](imInfoData), storageOffset = 1, sizes = Array(i, 4))
          }
          if (convertLabel) {
            labelTensor.set(Storage[Float](labelData.toArray),
              storageOffset = 1, sizes = Array(labelData.length / 7, 7))
          }
          MiniBatch(featureTensor, labelTensor, imInfoTensor)
        } else {
          null
        }
      }
    }
  }
}


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
package com.intel.analytics.zoo.models.dataset

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, Transformer, Utils}
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.Iterator


class SampleToBatch(totalBatch: Int, partitionNum: Option[Int] = None)
  extends Transformer[ImageSample, ImageBatch]{
  private val batchPerPartition = Utils.getBatchSize(totalBatch, partitionNum)
  override def apply(prev: Iterator[ImageSample]): Iterator[ImageBatch] = {
    val batchSizePerPartition = batchPerPartition
    new Iterator[ImageBatch] {

      private val batchSize = batchSizePerPartition
      private val sampleData = new Array[ImageSample](batchSize)
      override def hasNext: Boolean = prev.hasNext
      override def next(): ImageBatch = {
        if (prev.hasNext) {
          var i = 0
          val imgInfo = Seq[String]()
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            sampleData(i) = sample
            imgInfo :+ sample.infor
            i += 1
          }

          if (i < batchSize) {
            ImageBatch(sampleData.slice(0, i))
          } else {
            ImageBatch(sampleData)
          }

        } else {
          null
        }
      }
    }
  }
}

object SampleToBatch {
  def apply(totalBatch: Int, partitionNum: Option[Int] = None): SampleToBatch = new SampleToBatch(totalBatch, partitionNum)
}

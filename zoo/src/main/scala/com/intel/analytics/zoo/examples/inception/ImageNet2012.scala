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

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{BGRImgCropper, BGRImgNormalizer, BytesToBGRImg, MTLabeledBGRImgToBatch, HFlip => DatasetHFlip}
import com.intel.analytics.zoo.feature.{DistributedFeatureSet, FeatureSet}
import com.intel.analytics.zoo.feature.pmem.{DRAM, MemoryType, PMEM}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext

object ImageNet2012 {

  /**
   * Extract hadoop sequence files from an HDFS path
   * @param url
   * @param sc
   * @param classNum
   * @return
   */
  private def readFromSeqFiles(url: String, sc: SparkContext, classNum: Int) = {
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
    memoryType: MemoryType = DRAM
  )
  : DataSet[MiniBatch[Float]] = {
    val rawData = readFromSeqFiles(path, sc, classNumber)
    val featureSet = FeatureSet.rdd(rawData, memoryType = memoryType)
    featureSet.transform(
      MTLabeledBGRImgToBatch[ByteRecord](
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = (BytesToBGRImg() -> BGRImgCropper(imageSize, imageSize)
          -> DatasetHFlip(0.5) -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225))
      ))
  }
}

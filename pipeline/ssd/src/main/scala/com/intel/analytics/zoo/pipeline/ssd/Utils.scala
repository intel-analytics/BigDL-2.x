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

package com.intel.analytics.zoo.pipeline.ssd

import java.io.File

import com.intel.analytics.zoo.pipeline.common.dataset.LocalByteRoiimageReader
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage._
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


object IOUtils {
  def loadSeqFiles(nPartition: Int, seqFloder: String, sc: SparkContext,
    hasLabel: Boolean = false): RDD[RoiByteImage] = {
    val recordToByteImageWithRoi = RecordToByteRoiImage(hasLabel)
    val data = sc.sequenceFile(seqFloder, classOf[Text], classOf[Text],
      nPartition).map(x => SSDByteRecord(x._2.copyBytes(), x._1.toString))
    data.mapPartitions(recordToByteImageWithRoi(_))
  }

  def loadLocalFolder(nPartition: Int, folder: String, sc: SparkContext): RDD[RoiByteImage] = {
    val roiDataset = localImagePaths(folder).map(RoiImagePath(_))
    val imgReader = LocalByteRoiimageReader()
    sc.parallelize(roiDataset.map(roidb => imgReader.transform(roidb)),
      nPartition)
  }

  def localImagePaths(folder: String): Array[String] = {
    new File(folder).listFiles().map(_.getAbsolutePath)
  }
}

case class PreProcessParam(batchSize: Int = 4,
                           resolution: Int = 300,
                           pixelMeanRGB: (Float, Float, Float),
                           hasLabel: Boolean
                          )

case class PostProcessParam(nClasses: Int = 21, shareLocation: Boolean = true, bgLabel: Int = 0,
                            nmsThresh: Float = 0.45f, nmsTopk: Int = 400, keepTopK: Int = 200, confThresh: Float = 0.01f,
                            varianceEncodedInTarget: Boolean = false, useDiff: Boolean = true)

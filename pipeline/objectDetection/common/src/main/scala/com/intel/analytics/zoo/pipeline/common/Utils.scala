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

package com.intel.analytics.zoo.pipeline.common

import java.io.File

import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage._
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


object IOUtils {

  def loadSeqFiles(nPartition: Int, seqFloder: String, sc: SparkContext): RDD[ByteRecord] = {
    sc.sequenceFile(seqFloder, classOf[Text], classOf[Text],
      nPartition).map(x => ByteRecord(x._2.copyBytes(), x._1.toString))
  }

  def localImagePaths(folder: String): LocalImageFrame = {
    val arr = new File(folder).listFiles().map(x => {
      val imf = ImageFeature()
      imf(ImageFeature.uri) = x.getAbsolutePath
      imf
    })
    ImageFrame.array(arr)
  }
}


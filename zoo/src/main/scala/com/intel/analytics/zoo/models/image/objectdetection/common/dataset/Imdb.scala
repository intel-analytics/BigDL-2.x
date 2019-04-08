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

package com.intel.analytics.zoo.models.image.objectdetection.common.dataset

import java.io.File
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.feature.image.{ImageSet, LocalImageSet}
import com.intel.analytics.zoo.feature.image.roi.RoiRecordToFeature
import org.apache.commons.io.FileUtils
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

trait Imdb {
  def getRoidb(readImage: Boolean = true): LocalImageSet

  def loadImage(imagePath: String): Array[Byte] = {
    FileUtils.readFileToByteArray(new File(imagePath))
  }
}


object Imdb {
  /**
   * Get an imdb (image database) by name
   *
   * @param name
   * @param devkitPath
   * @return
   */
  def getImdb(name: String, devkitPath: String): Imdb = {
    val items = name.split("_")
    if (items(0) == "voc") {
      new PascalVoc(items(1), items(2), devkitPath)
    } else if (items(0) == "coco") {
      new Coco(items(1), devkitPath)
    } else {
      new CustomizedDataSet(name, devkitPath)
    }
  }


  def data(roidb: Array[ImageFeature]): Iterator[ImageFeature] = {
    new Iterator[ImageFeature] {
      private val index = new AtomicInteger()

      override def hasNext: Boolean = {
        index.get() < roidb.length
      }

      override def next(): ImageFeature = {
        val curIndex = index.getAndIncrement()
        if (curIndex < roidb.length) {
          roidb(curIndex)
        } else {
          null.asInstanceOf[ImageFeature]
        }
      }
    }
  }

  def loadRoiSeqFiles(seqFloder: String, sc: SparkContext, nPartition: Option[Int] = None):
  RDD[ByteRecord] = {
    val raw = if (nPartition.isDefined) {
      sc.sequenceFile(seqFloder, classOf[Text], classOf[Text], nPartition.get)
    } else {
      sc.sequenceFile(seqFloder, classOf[Text], classOf[Text])
    }
    raw.map(x => ByteRecord(x._2.copyBytes(), x._1.toString))
  }

  def roiSeqFilesToImageSet(url: String, sc: SparkContext, partitionNum: Option[Int] = None):
  ImageSet = {
    val rdd = loadRoiSeqFiles(url, sc, partitionNum)
    val featureRDD = RoiRecordToFeature(true)(rdd)
    ImageSet.rdd(featureRDD)
  }
}

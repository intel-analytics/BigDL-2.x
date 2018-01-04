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

package com.intel.analytics.zoo.pipeline.common.dataset

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.RoiImagePath

import scala.io.Source

trait Imdb {
  def loadRoidb(): Array[RoiImagePath]
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


  def data(roidb: Array[RoiImagePath]): Iterator[RoiImagePath] = {
    new Iterator[RoiImagePath] {
      private val index = new AtomicInteger()

      override def hasNext: Boolean = {
        index.get() < roidb.length
      }

      override def next(): RoiImagePath = {
        val curIndex = index.getAndIncrement()
        if (curIndex < roidb.length) {
          roidb(curIndex)
        } else {
          null.asInstanceOf[RoiImagePath]
        }
      }
    }
  }
}

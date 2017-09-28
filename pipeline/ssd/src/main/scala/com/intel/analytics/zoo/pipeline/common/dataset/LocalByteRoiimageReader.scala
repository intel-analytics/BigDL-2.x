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

import java.io.File
import java.nio.ByteBuffer

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.{RoiByteImageToSeq, RoiImagePath, SSDByteRecord}
import org.apache.commons.io.FileUtils

/**
 * load local image and target if exists
 */
class LocalByteRoiimageReader extends Transformer[RoiImagePath, SSDByteRecord] {
  private val preBuffer: ByteBuffer = ByteBuffer.allocate(4 * 2)

  override def apply(prev: Iterator[RoiImagePath]): Iterator[SSDByteRecord] = {
    prev.map(data => {
      transform(data)
    })
  }

  def transform(roiImagePath: RoiImagePath, useBuffer: Boolean = true): SSDByteRecord = {
    val imageInByte = FileUtils.readFileToByteArray(new File(roiImagePath.imagePath))
    preBuffer.putInt(imageInByte.length)
    val classLen = if (roiImagePath.target != null) roiImagePath.target.classes.size(2) * 4 else 0
    preBuffer.putInt(classLen)
    val data: Array[Byte] = new Array[Byte](preBuffer.capacity + imageInByte.length +
      classLen * 6 * 4)
    var startInd = 0
    System.arraycopy(preBuffer.array, 0, data, 0, preBuffer.capacity)
    startInd += preBuffer.capacity
    System.arraycopy(imageInByte, 0, data, startInd, imageInByte.length)
    startInd += imageInByte.length
    if (roiImagePath.target != null) {
      val cls = RoiByteImageToSeq.tensorToBytes(roiImagePath.target.classes)
      System.arraycopy(cls, 0, data, startInd, cls.length)
      startInd += cls.length
      val bbox = RoiByteImageToSeq.tensorToBytes(roiImagePath.target.bboxes)
      System.arraycopy(bbox, 0, data, startInd, bbox.length)
    }
    preBuffer.clear
    SSDByteRecord(data, roiImagePath.imagePath)
  }
}

object LocalByteRoiimageReader {
  def apply(): LocalByteRoiimageReader = new LocalByteRoiimageReader()
}

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

import java.awt.image.BufferedImage
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, FileInputStream}
import java.nio.channels.Channels
import java.nio.file.{Path, Paths}
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.BGRImage
import com.intel.analytics.zoo.pipeline.common.dataset.roiimage.{RoiByteImage, RoiImagePath}

/**
 * load local image and target if exists
 */
class LocalByteRoiimageReader extends Transformer[RoiImagePath, RoiByteImage] {
  override def apply(prev: Iterator[RoiImagePath]): Iterator[RoiByteImage] = {
    prev.map(data => {
      transform(data)
    })
  }

  def transform(roiImagePath: RoiImagePath, useBuffer: Boolean = true): RoiByteImage = {
    val originalImage = LocalByteRoiimageReader.readRawImage(Paths.get(roiImagePath.imagePath))
    // convert BufferedImage to byte array
    val baos = new ByteArrayOutputStream()
    // use jpg will lost some information
    ImageIO.write(originalImage, "png", baos)
    baos.flush()
    val imageInByte = baos.toByteArray
    baos.close()
    RoiByteImage(imageInByte, imageInByte.length, roiImagePath.imagePath, roiImagePath.target)
  }
}

object LocalByteRoiimageReader {
  def apply(): LocalByteRoiimageReader = new LocalByteRoiimageReader()

  def readRawImage(path: Path): BufferedImage = {
    var fis: FileInputStream = null
    try {
      fis = new FileInputStream(path.toString)
      val channel = fis.getChannel
      val byteArrayOutputStream = new ByteArrayOutputStream
      channel.transferTo(0, channel.size, Channels.newChannel(byteArrayOutputStream))
      val image = ImageIO.read(new ByteArrayInputStream(byteArrayOutputStream.toByteArray))
      require(image != null, "Can't read file " + path + ", ImageIO.read is null")
      image
    } catch {
      case ex: Exception =>
        ex.printStackTrace()
        System.err.println("Can't read file " + path)
        throw ex
    } finally {
      if (fis != null) {
        fis.close()
      }
    }
  }
}

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


package com.intel.analytics.zoo.transform.vision.feature

import java.awt.Color
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, FileInputStream}
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.file.Path
import javax.imageio.ImageIO

/**
 * original BigDL image reading code. Use for result comparison.
 */
object BigDLBGRImage {
  def hflip(data : Array[Float], height : Int, width : Int): Unit = {
    var y = 0
    while (y < height) {
      var x = 0
      while (x < width / 2) {
        var swap = 0.0f
        swap = data((y * width + x) * 3)
        data((y * width + x) * 3) = data((y * width + width - x - 1) * 3)
        data((y * width + width - x - 1) * 3) = swap

        swap = data((y * width + x) * 3 + 1)
        data((y * width + x) * 3 + 1) = data((y * width + width - x - 1) * 3 + 1)
        data((y * width + width - x - 1) * 3 + 1) = swap

        swap = data((y * width + x) * 3 + 2)
        data((y * width + x) * 3 + 2) = data((y * width + width - x - 1) * 3 + 2)
        data((y * width + width - x - 1) * 3 + 2) = swap
        x += 1
      }
      y += 1
    }
  }

  private def getWidthHeightAfterRatioScale(oriHeight: Int, oriWidth: Int,
                                            scaleTo: Int): (Int, Int) = {
    if (NO_SCALE == scaleTo) {
      (oriHeight, oriWidth)
    } else {
      if (oriWidth < oriHeight) {
        (scaleTo * oriHeight / oriWidth, scaleTo)
      } else {
        (scaleTo, scaleTo * oriWidth / oriHeight)
      }
    }
  }

  private def readRawImage(path: Path): BufferedImage = {
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

  val NO_SCALE = -1

  def resizeImage(img: BufferedImage, resizeWidth: Int, resizeHeight: Int): Array[Byte] = {
    var scaledImage: java.awt.Image = null
    // no scale
    if ((resizeHeight == img.getHeight) && (resizeWidth == img.getWidth)) {
      scaledImage = img
    } else {
      scaledImage =
        img.getScaledInstance(resizeWidth, resizeHeight, java.awt.Image.SCALE_AREA_AVERAGING)
    }

    val imageBuff: BufferedImage =
      new BufferedImage(resizeWidth, resizeHeight, BufferedImage.TYPE_3BYTE_BGR)
    imageBuff.getGraphics.drawImage(scaledImage, 0, 0, new Color(0, 0, 0), null)
    val pixels: Array[Byte] =
      imageBuff.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    require(pixels.length % 3 == 0)

    val bytes = new Array[Byte](8 + pixels.length)
    val byteBuffer = ByteBuffer.wrap(bytes)
    require(imageBuff.getWidth * imageBuff.getHeight * 3 == pixels.length)
    byteBuffer.putInt(imageBuff.getWidth)
    byteBuffer.putInt(imageBuff.getHeight)
    System.arraycopy(pixels, 0, bytes, 8, pixels.length)
    bytes
  }

  def readImage(path: Path, scaleTo: Int): Array[Byte] = {
    val img: BufferedImage = readRawImage(path)
    val (heightAfterScale, widthAfterScale) =
      getWidthHeightAfterRatioScale(img.getHeight, img.getWidth, scaleTo)
    resizeImage(img, widthAfterScale, heightAfterScale)
  }

  def readImage(path: Path, resizeWidth: Int, resizeHeight: Int): Array[Byte] = {
    val img: BufferedImage = readRawImage(path)
    resizeImage(img, resizeWidth, resizeHeight)
  }
}

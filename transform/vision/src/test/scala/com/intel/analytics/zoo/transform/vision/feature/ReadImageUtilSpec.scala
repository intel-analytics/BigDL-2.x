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

package com.intel.analytics.zoo.transform.vision.feature;

import java.awt.Color
import java.awt.image.DataBufferByte
import java.io.{ByteArrayOutputStream, File, FileInputStream, InputStream}
import java.nio.file.Paths
import javax.imageio.ImageIO

import com.intel.analytics.zoo.transform.vision.image.feature.ReadImageUtil
import org.apache.commons.io.FileUtils
import org.opencv.imgcodecs.Imgcodecs
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class ReadImageUtilSpec extends FlatSpec with Matchers with BeforeAndAfterAll{

  val resourcePath = "image/ILSVRC2012_val_00000003.JPEG"
  override def beforeAll(): Unit = {
//    OpenCV.load()
  }

//  "read as Mat" should "work properly" in {
//    OpenCV.load()
//    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
//    val bytes = FileUtils.readFileToByteArray(Paths.get(resource.toURI()).toFile)
//    val img = ReadImageUtil.readImageAsMat(bytes)
//    val tmpFile = java.io.File.createTempFile("module", ".jpg")
//    Imgcodecs.imwrite(tmpFile.getAbsolutePath, img)
//    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
//  }

  "read as bytes" should "get same result as ImageIO" in {
    val resource = getClass.getClassLoader.getResource(resourcePath)
    val fileBytes = FileUtils.readFileToByteArray(Paths.get(resource.toURI).toFile)
    val (bytes, h, w, c) = ReadImageUtil.readImageAsBytes(fileBytes)
    val (bytes2, h2, w2, c2) = ReadImageUtilSpec.getBytesFromStream(Paths.get(resource.toURI).toFile)
    val matchingPoints = bytes.zip(bytes2).count(t => Math.abs(t._1 - t._2) < 1)
    assert(matchingPoints > bytes.length * 0.99)
    assert((h, w, c) == (h2, w2, c2))
    assert(bytes.length == 500 * 375 * 3)
  }

  "read as bytes" should "get same result as BigDL ImageIO with smallSideSize" in {
    val resource = getClass.getClassLoader.getResource(resourcePath)
    val fileBytes = FileUtils.readFileToByteArray(Paths.get(resource.toURI).toFile)
    // Mat decoding
    val (bytes, h, w, c) = ReadImageUtil.readImageAsBytes(fileBytes, 256)

    // ImageIO and rescale
    val bytes2 = BigDLBGRImage.readImage(Paths.get(resource.toURI), 256).drop(8)
    val matchingPoints = bytes.zip(bytes2).count(t => Math.abs(t._1 - t._2) <= 5)
    println(matchingPoints * 1.0 / bytes.length)
    assert(matchingPoints > bytes.length * 0.99)
    assert(bytes.length == 341 * 256 * 3)
  }

  "read as Mat" should "take smallSideSize" in {
    val resource = getClass.getClassLoader.getResource(resourcePath)
    val fileBytes = FileUtils.readFileToByteArray(Paths.get(resource.toURI).toFile)
    val (bytes, h, w, c) = ReadImageUtil.readImageAsBytes(fileBytes, 256)
    assert(bytes.length == h * w * c)
    assert(h == 256 && w == 500 * 256 / 375)
  }

  "read as Bytes" should "take smallSideSize" in {
    val resource = getClass.getClassLoader.getResource(resourcePath)
    val fileBytes = FileUtils.readFileToByteArray(Paths.get(resource.toURI).toFile)
    val mat = ReadImageUtil.readImageAsMat(fileBytes, 256)
    assert(mat.height() == 256 && mat.width() == 500 * 256 / 375)
  }

  "read as Floats" should "take smallSideSize and divisor" in {
    val resource = getClass.getClassLoader.getResource(resourcePath)
    val fileBytes = FileUtils.readFileToByteArray(Paths.get(resource.toURI).toFile)
    val (floats, h, w, c) = ReadImageUtil.readImageAsFloats(fileBytes, 256, 255.0f)
    assert(floats.length == h * w * c)
    assert(floats.forall(f => f <= 1.0 && f >= 0.0))
  }

}

object ReadImageUtilSpec {

  // BGR
  private def getBytesFromStream(file: File): (Array[Byte], Int, Int, Int) = {
    val img = ImageIO.read(file)

    val height = img.getHeight
    val width = img.getWidth
    val nChannels = if (img.getColorModel().hasAlpha()) 4 else 3

    assert(height * width * nChannels < 1e9, "image is too large")
    val decoded = Array.ofDim[Byte](height * width * nChannels)

    var offset = 0
    for (h <- 0 until height) {
      for (w <- 0 until width) {
        val color = new Color(img.getRGB(w, h))

        decoded(offset) = color.getBlue.toByte
        decoded(offset + 1) = color.getGreen.toByte
        decoded(offset + 2) = color.getRed.toByte
        if (nChannels == 4) {
          decoded(offset + 3) = color.getAlpha.toByte
        }
        offset += nChannels
      }
    }

    (decoded, img.getHeight, img.getWidth, nChannels)
  }


}

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
import java.io.File
import java.nio.file.Paths
import javax.imageio.ImageIO

import com.intel.analytics.zoo.transform.vision.image.feature.{BGRImageReader, ReadImageUtil}
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCV
import org.apache.commons.io.{FileUtils, IOUtils}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class ImageReaderSpec extends FlatSpec with Matchers with BeforeAndAfterAll {

  var spark: SparkSession = _
  val resource = getClass.getClassLoader.getResource("image/ILSVRC2012_val_00000003.JPEG")

  override def beforeAll(): Unit = {
    OpenCV.loadIfNecessary()
    spark = SparkSession.builder().master("local[1]").appName("test").getOrCreate()
  }

  "BGRImageReader" should "transform from path" in {
    val sp = this.spark
    import sp.implicits._
    val pathDF = Seq(Paths.get(resource.toURI).toAbsolutePath.toString,
      Paths.get(resource.toURI).toAbsolutePath.toString).toDF("p")
    val imageDF = new BGRImageReader().setInputCol("p").setOutputCol("imageData").transform(pathDF)

    assert(imageDF.columns.sameElements(Array ("p", "imageData")))
    assert(imageDF.select("imageData").collect().forall {
      case Row(Row(bytes: Array[Byte], h: Int, w: Int, c: Int)) =>
        bytes.length == h * w * c && h == 256 && w == 341 && c == 3
    })
    imageDF.printSchema()
  }

  "BGRImageReader" should "read image with correct dimension" in {
    val imageDF = BGRImageReader.readImagesToBytes(
      Paths.get(resource.toURI).toAbsolutePath.toString, spark, 256)

    assert(imageDF.columns.sameElements(Array ("path", "imageData")))
    assert(imageDF.select("imageData").collect().forall {
      case Row(Row(bytes: Array[Byte], h: Int, w: Int, c: Int)) =>
        bytes.length == h * w * c && h == 256 && w == 341 && c == 3
    })
  }

  "BGRImageReader" should "get the same result as ReadImageUtil" in {
    // DF Interface
    val imageDF = BGRImageReader.readImagesToBytes(
      Paths.get(resource.toURI).toAbsolutePath.toString, spark, 256)

    // JVM Interface
    val src = new Path(resource.getPath)
    val fs = src.getFileSystem(new Configuration())
    val is = fs.open(src)
    val fileBytes = IOUtils.toByteArray(is)
    val bytes2 = ReadImageUtil.readImageAsBytes(fileBytes, 256)._1

    // should 100% match
    assert(imageDF.select("imageData").collect().forall {
      case Row(Row(bytes: Array[Byte], h: Int, w: Int, c: Int)) =>
        bytes.zip(bytes2).forall(t => t._1 == t._2)
    })
  }

  "Spark Binary file" should "get same bytes as Apache IO" in {
    val resource = getClass.getClassLoader.getResource("image/ILSVRC2012_val_00000003.JPEG")
    // DF Interface
    val fileBytesRDD = spark.sparkContext.binaryFiles(resource.getFile)
    val fileBytes = FileUtils.readFileToByteArray(Paths.get(resource.toURI).toFile)

    fileBytesRDD.collect().foreach { t =>
      val sparkBytes = t._2.toArray()
      assert (fileBytes.zip(sparkBytes).forall( p => p._1 == p._2))
    }
  }

  "BGRImageReader" should "get similar result as direct Image IO" in {
    val resource = getClass.getClassLoader.getResource("image/ILSVRC2012_val_00000003.JPEG")
    val imageDF = BGRImageReader.readImagesToBytes(
      Paths.get(resource.toURI).toAbsolutePath.toString, spark)

    val (bytes2, h2, w2, c2) = ImageReaderSpec.getBytesFromStream(Paths.get(resource.toURI).toFile)

    assert(imageDF.columns.sameElements(Array ("path", "imageData")))
    assert(imageDF.select("imageData").collect().forall {
      case Row(Row(bytes: Array[Byte], h: Int, w: Int, c: Int)) =>
        val matchingPoints = bytes.zip(bytes2).count(t => Math.abs(t._1 - t._2) < 5)
        println(matchingPoints)
        println(bytes.length)
        matchingPoints > bytes.length * 0.99
    })
  }

  "BGRImageReader" should "get similar result as BigDL ImageIO" in {
    // DF Interface
    val imageDF = BGRImageReader.readImagesToBytes(
      Paths.get(resource.toURI).toAbsolutePath.toString, spark, 256)
    assert(imageDF.columns.sameElements(Array ("path", "imageData")))
    val dfBytes = imageDF.select("imageData").head().getAs[Row](0).getAs[Array[Byte]](0)

    // BigDL ImageIO
    val bigDLBytes = BigDLBGRImage.readImage(Paths.get(resource.toURI), 256).drop(8)

    assert(ImageReaderSpec.bytesMatch(dfBytes, bigDLBytes))
  }

  "ImageReader" should "read image as Floats" in {
    val resource = getClass.getClassLoader.getResource("image/ILSVRC2012_val_00000003.JPEG")
    val imageDF = BGRImageReader.readImagesAsFloats(
      Paths.get(resource.toURI).toAbsolutePath.toString, spark, 256)

    val h = imageDF.head()
    assert(imageDF.columns.sameElements(Array ("path", "imageData")))
    val floats = imageDF.select("imageData").rdd.map { r =>
      val data = r.getAs[Row](0).getSeq[Float](0)
      data
    }.collect()

    assert(floats.forall(_.length == 256 * 341 * 3))
  }

}

object ImageReaderSpec {

  def bytesMatch(bytes1: Array[Byte], bytes2: Array[Byte]): Boolean = {
    val matchingPoints = bytes1.zip(bytes2).count(t => Math.abs(t._1 - t._2) <= 5)
    println(matchingPoints * 1.0 / bytes2.length)
    matchingPoints > bytes2.length * 0.99
  }

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


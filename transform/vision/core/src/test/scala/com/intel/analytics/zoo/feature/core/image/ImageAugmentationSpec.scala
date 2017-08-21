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

package com.intel.analytics.zoo.feature.core.image

import java.io.File
import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.zoo.feature.core.label.roi._
import com.intel.analytics.zoo.feature.core.util.{BboxUtil, MatWrapper, NormalizedBox}
import org.opencv.core.{Mat, Point, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.scalatest.{FlatSpec, Matchers}

class ImageAugmentationSpec extends FlatSpec with Matchers {
  "resize" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val out = Resize.transform(img, img, 300, 300)
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, out)
    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
  }

  "colorjitter" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val feature = Feature()
    feature(Feature.mat) = img
    val colorJitter = ColorJitter()
    val out = colorJitter(feature)
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, img)
    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
  }

  "hflip" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val out = HFlip.transform(img, img)
    val name = s"input000025" +
      s"hflip-${System.nanoTime()}"
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, out)
    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
  }

  "expand" should "work properly" in {
    import scala.sys.process._
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val feature = new Feature
    feature(Feature.mat) = img
    val expand = new Expand()
    expand(feature)
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, feature.inputMat())
    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
  }

  "expand with roi" should "work properly" in {
    import scala.sys.process._
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val feature = new Feature
    val classes = Array(11.0, 11.0, 11.0, 16.0, 16.0, 16.0, 11.0, 16.0,
      16.0, 16.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0).map(_.toFloat) //)).resize(2, 11)
    val boxes = Array(2.0, 84.0, 59.0, 248.0,
      68.0, 115.0, 233.0, 279.0,
      64.0, 173.0, 377.0, 373.0,
      320.0, 2.0, 496.0, 375.0,
      221.0, 4.0, 341.0, 374.0,
      135.0, 14.0, 220.0, 148.0,
      69.0, 43.0, 156.0, 177.0,
      58.0, 54.0, 104.0, 139.0,
      279.0, 1.0, 331.0, 86.0,
      320.0, 22.0, 344.0, 96.0,
      337.0, 1.0, 390.0, 107.0).map(_.toFloat) //)).resize(11, 4)
    val label = RoiLabel(Tensor(Storage(classes)).resize(2, 11),
      Tensor(Storage(boxes)).resize(11, 4))
    feature(Feature.mat) = img
    feature(Feature.label) = label
    val expand = RoiNormalize() -> Expand() -> RoiExpand()
    expand(feature)
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    visulize(label, img)
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, feature.inputMat())
    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
  }

  "expand with roi random" should "work properly" in {
    import scala.sys.process._
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val feature = new Feature
    val classes = Array(11.0, 11.0, 11.0, 16.0, 16.0, 16.0, 11.0, 16.0,
      16.0, 16.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0).map(_.toFloat) //)).resize(2, 11)
    val boxes = Array(2.0, 84.0, 59.0, 248.0,
      68.0, 115.0, 233.0, 279.0,
      64.0, 173.0, 377.0, 373.0,
      320.0, 2.0, 496.0, 375.0,
      221.0, 4.0, 341.0, 374.0,
      135.0, 14.0, 220.0, 148.0,
      69.0, 43.0, 156.0, 177.0,
      58.0, 54.0, 104.0, 139.0,
      279.0, 1.0, 331.0, 86.0,
      320.0, 22.0, 344.0, 96.0,
      337.0, 1.0, 390.0, 107.0).map(_.toFloat) //)).resize(11, 4)
    val label = RoiLabel(Tensor(Storage(classes)).resize(2, 11),
      Tensor(Storage(boxes)).resize(11, 4))
    feature(Feature.mat) = img
    feature(Feature.label) = label
    val expand = RoiNormalize() -> new RandomOp(Expand() -> RoiExpand()
      , 0.5)
    expand(feature)
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    visulize(label, img)
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, feature.inputMat())
    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
  }

  "crop" should "work properly" in {
    import scala.sys.process._
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    Crop.transform(img, img, NormalizedBox(0, 0f, 1, 0.5f))
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, img)
    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
  }

  "brightness" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val out = Brightness.transform(img, img, 32)
    val name = s"input000025" +
      s"brightness-${System.nanoTime()}"
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, out)
    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
  }

  "ChannelOrder" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    ChannelOrder.transform(img, img)
    val name = s"input000025" +
      s"ChannelOrder-${System.nanoTime()}"
    val tmpFile = java.io.File.createTempFile(name, ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, img)
    println(s"save to ${tmpFile.getAbsolutePath}, " + new File(tmpFile.getAbsolutePath).length())
  }

  "Normalize" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val buf = new MatWrapper()
    val oriFloats = new Array[Float](img.height() * img.width() * 3)
    MatWrapper.toFloatBuf(img, oriFloats, buf)
    Normalize.transform(img, img, 100, 200, 300)
    val floats = new Array[Float](img.height() * img.width() * 3)
    MatWrapper.toFloatBuf(img, floats, buf)

    var i = 1
    oriFloats.zip(floats).foreach(x => {
      if (i % 3 == 1) assert(x._1 - x._2 == 300)
      if (i % 3 == 2) assert(x._1 - x._2 == 200)
      if (i % 3 == 3) assert(x._1 - x._2 == 100)
      i += 1
    })

  }

  "Normalize new output" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val out = new MatWrapper()
    val buf = new MatWrapper()
    val oriFloats = new Array[Float](img.height() * img.width() * 3)
    MatWrapper.toFloatBuf(img, oriFloats, buf)
    Normalize.transform(img, out, 100, 200, 300)
    val floats = new Array[Float](img.height() * img.width() * 3)
    MatWrapper.toFloatBuf(out, floats, buf)

    var i = 1
    oriFloats.zip(floats).foreach(x => {
      if (i % 3 == 1) assert(x._1 - x._2 == 300)
      if (i % 3 == 2) assert(x._1 - x._2 == 200)
      if (i % 3 == 3) assert(x._1 - x._2 == 100)
      i += 1
    })

  }


  "ImageAugmentation" should "work properly" in {
    import scala.sys.process._
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = Files.readAllBytes(Paths.get(resource.getFile))
    val features = Array(Feature(img)).toIterator
    val imgAug = BytesToMat() ->
      ColorJitter() ->
      Expand() ->
      Resize(300, 300, -1) ->
      HFlip() ->
      Normalize((123, 117, 104)) ->
      new MatToFloats()
    val out = imgAug.toIterator(features)
    out.foreach(img => {
      val tmpFile = java.io.File.createTempFile("module", ".jpg")
      val mat = MatWrapper.floatToMat(img.getFloats(), img.getHeight(), img.getWidth())
      Imgcodecs.imwrite(tmpFile.getAbsolutePath, mat)
      println(s"save to ${tmpFile.getAbsolutePath}, "
        + new File(tmpFile.getAbsolutePath).length())
    })
  }

  def visulize(label: RoiLabel, mat: Mat): Unit = {
    var i = 1
    while (label.bboxes.nElement() > 0 && i <= label.bboxes.size(1)) {
      Imgproc.rectangle(mat, new Point(label.bboxes.valueAt(i, 1) * mat.width(),
        label.bboxes.valueAt(i, 2) * mat.height()),
        new Point(label.bboxes.valueAt(i, 3) * mat.width(),
          label.bboxes.valueAt(i, 4) * mat.height()),
        new Scalar(0, 255, 0))
      i += 1
    }
  }

  "mat encode byte array" should "work" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = Files.readAllBytes(Paths.get(resource.getFile))
    val bytes = new Array[Byte](2 * img.length)

    img.copyToArray(bytes, 0)

    val mat = MatWrapper.toMat(bytes)
    val mat2 = MatWrapper.toMat(img)

    println(mat.height(), mat.width())
    println(mat2.height(), mat2.width())
    val floats1 = new Array[Float](375 * 500 * 3)
    val floats2 = new Array[Float](375 * 500 * 3)
    val buf = new MatWrapper()
    val n1 = MatWrapper.toFloatBuf(mat, floats1, buf)
    val n2 = MatWrapper.toFloatBuf(mat2, floats2, buf)
    floats1.zip(floats2).foreach(x => assert(x._1 == x._2))
  }


  "Image Transformer with exception" should "work properly" in {
    import scala.sys.process._
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = Files.readAllBytes(Paths.get(resource.getFile))
    val byteImage = Feature(img)
    val imgAug = BytesToMat() ->
      Crop(useNormalized = false, bbox = Some(NormalizedBox(-1, -1, -1, -1))) ->
      Resize(300, 300, -1) ->
      MatToFloats()
    val out = imgAug(byteImage)
    out.getFloats().length should be(3 * 300 * 300)
  }

  "Image Transformer with empty byte input" should "work properly" in {
    val img = Array[Byte]()
    val byteImage = Feature(img)
    val imgAug = BytesToMat() ->
      Resize(1, 1, -1) ->
      Crop(useNormalized = false, bbox = Some(NormalizedBox(-1, -1, -1, -1))) ->
      MatToFloats()
    val out = imgAug(byteImage)
    out.getFloats().length should be(3)
  }
}

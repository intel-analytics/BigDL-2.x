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

package com.intel.analytics.dlfeature.core.image

import java.io.File
import java.nio.file.{Files, Paths}

import com.intel.analytics.dlfeature.core.label.roi._
import com.intel.analytics.dlfeature.core.util.{MatWrapper, NormalizedBox}
import org.opencv.core.{Mat, Point, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.scalatest.{FlatSpec, Matchers}

import scala.sys.process._

class ImageAugmentationSpec extends FlatSpec with Matchers{
  "resize" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val out = Resize.transform(img, img, 300, 300)
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, out)
    println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
  }

  "colorjitter" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val out = ColorJitter.transform(img, img,
      0.5, 32, 0.5, 0.5, 1.5, 0.5, 18, 0.5, 0.5, 1.5, 0)
    val name = s"input000025" +
      s"colorJitter-${ System.nanoTime() }"
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, out)
    println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
  }

  "hflip" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val out = HFlip.transform(img, img)
    val name = s"input000025" +
      s"hflip-${ System.nanoTime() }"
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, out)
    println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
  }

  "expand" should "work properly" in {
    import scala.sys.process._
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val expand = new Expand(1)
    Expand.transform(img, 2, img)
    val name = s"input000025" +
      s"expand-${ System.nanoTime() }"
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, img)
    println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
  }

  "crop" should "work properly" in {
    import scala.sys.process._
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    Crop.transform(img, img, NormalizedBox(0, 0f, 1, 0.5f))
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, img)
    println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
  }

  "brightness" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    val out = Brightness.transform(img, img, 32)
    val name = s"input000025" +
      s"brightness-${ System.nanoTime() }"
    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, out)
    println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
  }

  "ChannelOrder" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
    val img = MatWrapper.read(resource.getFile)
    ChannelOrder.transform(img, img)
    val name = s"input000025" +
      s"ChannelOrder-${ System.nanoTime() }"
    val tmpFile = java.io.File.createTempFile(name, ".jpg")
    Imgcodecs.imwrite(tmpFile.getAbsolutePath, img)
    println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
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
    val byteImage = Array(ByteImage(img)).toIterator
    val imgAug = new BytesToFeature() ->
      BytesToMat() ->
      ColorJitter() ->
      Expand() ->
      Resize(300, 300, -1) ->
      HFlip() ->
      Normalize((123, 117, 104)) ->
      new MatToFloats()
    val out = imgAug(byteImage)
    out.foreach(img => {
      val tmpFile = java.io.File.createTempFile("module", ".jpg")
      val mat = MatWrapper.floatToMat(img.getFloats(), img.getHeight(), img.getWidth())
      Imgcodecs.imwrite(tmpFile.getAbsolutePath, mat)
      println(s"save to ${ tmpFile.getAbsolutePath }, "
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
    val byteImage = Array(ByteImage(img)).toIterator
    val imgAug = new BytesToFeature() ->
      BytesToMat() ->
      Crop(useNormalized = false, bbox = Some(NormalizedBox(-1, -1, -1, -1))) ->
      Resize(300, 300, -1) ->
      MatToFloats()
    val out = imgAug(byteImage)
    out.foreach(img => {
      img.getFloats().length should be (3 * 300 * 300)
    })
  }

  "Image Transformer with empty byte input" should "work properly" in {
    val img = Array[Byte]()
    val byteImage = Array(ByteImage(img)).toIterator
    val imgAug = new BytesToFeature() ->
      BytesToMat() ->
      Resize(1, 1, -1) ->
      Crop(useNormalized = false, bbox = Some(NormalizedBox(-1, -1, -1, -1))) ->
      MatToFloats()
    val out = imgAug(byteImage)
    out.foreach(img => {
      img.getFloats().length should be (3)
    })
  }


//  "ImageAugmentation with many loop" should "work properly" in {
//    val resource = getClass().getClassLoader().getResource("image/000025.jpg")
//    val img = Files.readAllBytes(Paths.get(resource.getFile))
//    val byteImage = Array(ByteImage(img)).toIterator.flatMap(x => {
//      (1 to 10000000).toIterator.map(i => x)
//    })
//    val imgAug = new BytesToFeature() ->
//      BytesToMat() ->
// //      ColorJitter() ->
// //      Expand() ->
//      Resize(300, 300, -1) ->
// //      HFlip() ->
//      Normalize((123, 117, 104)) ->
//      new MatToFloats()
//    val out = imgAug(byteImage)
//    out.foreach(img => {
// //      val tmpFile = java.io.File.createTempFile("module", ".jpg")
// //      val mat = MatWrapper.floatToMat(img.getFloats(), img.getHeight(), img.getWidth())
// //      Imgcodecs.imwrite(tmpFile.getAbsolutePath, mat)
// //      println(s"save to ${ tmpFile.getAbsolutePath }, "
// //        + new File(tmpFile.getAbsolutePath).length())
//    })
//  }


}

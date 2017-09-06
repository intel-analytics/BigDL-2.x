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

package com.intel.analytics.zoo.pipeline.common.dataset.roiimage

import java.io.File

import com.intel.analytics.zoo.pipeline.common.dataset.{Imdb, LocalByteRoiimageReader}
import com.intel.analytics.zoo.transform.vision.image.{BytesToMat, MatToFloats, RandomTransformer}
import com.intel.analytics.zoo.transform.vision.image.augmentation.{ColorJitter, Expand, HFlip, Resize}
import com.intel.analytics.zoo.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.transform.vision.label.roi._
import org.opencv.core.{Mat, Point, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

//
class DataAugmentationSpec extends FlatSpec with Matchers with BeforeAndAfter {
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

  "ImageAugmentation" should "work properly" in {
    import scala.sys.process._
    val resource = getClass().getClassLoader().getResource("VOCdevkit")
    val voc = Imdb.getImdb("voc_2007_testcode", resource.getPath)
    val roidb = voc.getRoidb().toIterator
    val imgAug = LocalByteRoiimageReader() -> RecordToFeature(true) ->
      BytesToMat() ->
      RoiNormalize() ->
      ColorJitter() ->
      RandomTransformer(Expand() -> RoiExpand(), 0.5) ->
      RandomSampler() ->
      Resize(300, 300, -1) ->
      RandomTransformer(HFlip() -> RoiHFlip(), 0.5) ->
      MatToFloats(validHeight = 300, validWidth = 300, meanRGB = Some(123, 117, 104))
    val out = imgAug(roidb)
    out.foreach(img => {
      val tmpFile = java.io.File.createTempFile("module", ".jpg")
      val mat = OpenCVMat.floatToMat(img.getFloats(), img.getHeight(), img.getWidth())
      visulize(img.getLabel[RoiLabel], mat)
      Imgcodecs.imwrite(tmpFile.getAbsolutePath, mat)
      println(s"save to ${tmpFile.getAbsolutePath}, "
        + new File(tmpFile.getAbsolutePath).length())
      s"display ${tmpFile.getAbsolutePath}" !
    })

  }
}

//
//  "train data preprocess" should "work properly" in {
//    val resource = getClass().getClassLoader().getResource("VOCdevkit")
//    val voc = Imdb.getImdb("voc_2007_testcode", resource.getPath)
//    val roidb = voc.getRoidb()
//    OpenCV.load()
//    roidb.foreach(img => {
//      val mat = Imgcodecs.imread(img.imagePath)
//      val originalHeight = mat.rows().toFloat
//      val originalWidth = mat.cols().toFloat
//
//      BboxUtil.scaleBBox(img.target.bboxes, 1 / originalHeight,
//        1 / originalWidth)
//      val dataAugmentation = DataAugmentation(300)
//      val cropedImage = dataAugmentation.process(mat, img.target)
//
//      visulize(img.target, cropedImage)
//      val name = s"input${ img.imagePath.substring(img.imagePath.lastIndexOf("/") + 1) }" +
//        s"final${ System.nanoTime() }"
//      val tmpFile = java.io.File.createTempFile("module", ".jpg")
//      Imgcodecs.imwrite(tmpFile.getAbsolutePath, cropedImage)
//      println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
//      println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
//      //      import scala.sys.process._
//      //      val cmd = s"display ${ tmpFile.getAbsolutePath }" // Your command
//      //      cmd.!! // Captures the output
//    })
//
//  }
//
//  "train data preprocess1" should "work properly" in {
//    val resource = getClass().getClassLoader().getResource("VOCdevkit")
//    val voc = Imdb.getImdb("voc_2007_testcode", resource.getPath)
//    val roidb = voc.getRoidb()
//    OpenCV.load()
//
//    val dataAugmentation = LocalByteRoiimageReader() ->
//      RecordToByteRoiImage(true) -> DataAugmentation(300)
//
//    val out = dataAugmentation.apply(roidb.toIterator)
//    out.foreach(img => {
//      val mat = new Mat(img.width, img.height, CvType.CV_32FC3)
//      mat.put(0, 0, img.data)
//      visulize(img.label.get.asInstanceOf[RoiLabel], mat)
//      val name = s"input${ img.path.substring(img.path.lastIndexOf("/") + 1) }" +
//        s"final${ System.nanoTime() }"
//      val tmpFile = java.io.File.createTempFile("module", ".jpg")
//      Imgcodecs.imwrite(tmpFile.getAbsolutePath, mat)
//      println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
//      import scala.sys.process._
//      val cmd = s"display ${ tmpFile.getAbsolutePath }" // Your command
//      cmd.!! // Captures the output
//    })
//
//  }
//
//  "train data preprocess2" should "work properly" in {
//    val resource = getClass().getClassLoader().getResource("VOCdevkit")
//    val voc = Imdb.getImdb("voc_2007_testcode", resource.getPath)
//    val roidb = voc.getRoidb()
//    OpenCV.load()
//
//    val dataAugmentation = LocalByteRoiimageReader() ->
//      RecordToByteRoiImage(true) -> new BytesToMatImage() ->
//      NormalizeRoi() ->
//      ColorJitter() ->
//      RoiImageExpand() ->
//      RandomSample() ->
//      Resize(300, 300, -1) ->
//      RoiImageHFlip() -> new MatImageToFloats()
//
//    val out = dataAugmentation.apply(roidb.toIterator)
//    out.foreach(img => {
//      val mat = new Mat(img.width, img.height, CvType.CV_32FC3)
//      mat.put(0, 0, img.data)
//      visulize(img.label.get.asInstanceOf[RoiLabel], mat)
//      val name = s"input${ img.path.substring(img.path.lastIndexOf("/") + 1) }" +
//        s"final${ System.nanoTime() }"
//      val tmpFile = java.io.File.createTempFile("module", ".jpg")
//      Imgcodecs.imwrite(tmpFile.getAbsolutePath, mat)
//      println(s"save to ${ tmpFile.getAbsolutePath }, " + new File(tmpFile.getAbsolutePath).length())
//      import scala.sys.process._
//      val cmd = s"display ${ tmpFile.getAbsolutePath }" // Your command
//      cmd.!! // Captures the output
//      s"rm ${ tmpFile.getAbsolutePath }".!!
//    })
//  }
//  var sc: SparkContext = null
//
//  Logger.getLogger("org").setLevel(Level.WARN)
//  Logger.getLogger("akka").setLevel(Level.WARN)
//  before {
//    val conf = Engine.createSparkConf().setMaster("local[2]")
//      .setAppName("BigDL SSD Demo")
//    sc = new SparkContext(conf)
//    Engine.init
//  }
//  "test data" should "work properly" in {
//    OpenCV.load()
//
//    val preProcessor =
//      RecordToByteRoiImage(false) ->
//        CVResizer(300, 300, resizeRois = false,
//          normalizeRoi = false) ->
//        RoiImageNormalizer((123f, 117f, 104f)) ->
//        RoiImageToBatch(2, false, Some(2))
//
//    val myTempDir = Files.createTempDirectory("test")
//    val tmpFile = java.io.File.createTempFile("module", ".jpg", myTempDir.toFile)
//    val mat1 = new Mat(0, 0, CvType.CV_8UC3)
//    Imgcodecs.imwrite(tmpFile.toString, mat1)
//
//    val tmpFile2 = java.io.File.createTempFile("module", ".jpg", myTempDir.toFile)
//    val mat2 = new Mat(1, 1, CvType.CV_8UC3)
//    Imgcodecs.imwrite(tmpFile2.toString, mat2)
//
//    val tmpFile3 = java.io.File.createTempFile("module", ".jpg", myTempDir.toFile)
//    val mat3 = new Mat(4, 4, CvType.CV_8UC3)
//    Imgcodecs.imwrite(tmpFile3.toString, mat3)
//
//    val tmpFile4 = java.io.File.createTempFile("module", ".jpg", myTempDir.toFile)
//    val mat4 = new Mat(40, 40, CvType.CV_8UC3)
//    Imgcodecs.imwrite(tmpFile4.toString, mat4)
//
//    val tmpFile5 = java.io.File.createTempFile("module", ".jpg", myTempDir.toFile)
//    val mat5 = new Mat(4000, 4000, CvType.CV_8UC3)
//    Imgcodecs.imwrite(tmpFile5.toString, mat5)
//
//    val data = IOUtils.loadLocalFolder(2, myTempDir.toAbsolutePath.toString, sc)
//    data._1.mapPartitions(preProcessor(_)).collect()
//
//    myTempDir.toFile.delete()
//  }
//
//
//  after {
//    if (sc != null) {
//      sc.stop()
//    }
//  }
//}

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

package com.intel.analytics.zoo.models.image.objectdetection.dataset.roiimage

import java.io.File

import com.intel.analytics.bigdl.transform.vision.image.label.roi._
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.Imdb
import org.opencv.core.{Mat, Point, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

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
    val roidb = voc.getRoidb(true).array.toIterator
    val imgAug = ImageBytesToMat() ->
      ImageRoiNormalize() ->
      ImageColorJitter() ->
      ImageRandomPreprocessing(ImageExpand() -> ImageRoiProject(), 0.5) ->
      ImageRandomSampler() ->
      ImageResize(300, 300, -1) ->
      ImageRandomPreprocessing(ImageHFlip() -> ImageRoiHFlip(), 0.5) ->
      ImageChannelNormalize(123f, 117f, 104f) ->
      ImageMatToFloats(validHeight = 300, validWidth = 300)
    val out = imgAug(roidb)
    out.foreach(img => {
      val tmpFile = java.io.File.createTempFile("module", ".jpg")
      val mat = OpenCVMat.fromFloats(img.floats(), img.getHeight(), img.getWidth())
      visulize(img.getLabel[RoiLabel], mat)
      Imgcodecs.imwrite(tmpFile.getAbsolutePath, mat)
      println(s"save to ${tmpFile.getAbsolutePath}, "
        + new File(tmpFile.getAbsolutePath).length())
    })

  }
}


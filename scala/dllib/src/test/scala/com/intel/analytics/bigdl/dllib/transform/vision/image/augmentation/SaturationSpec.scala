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

package com.intel.analytics.bigdl.transform.vision.image.augmentation

import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFeature, ImageFrame, LocalImageFrame}
import org.opencv.imgcodecs.Imgcodecs
import org.scalatest.{FlatSpec, Matchers}

class SaturationSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")
  "Saturation" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val transformer = Saturation(10, 20)
    val transformed = transformer(data)
    val imageFeature = transformed.asInstanceOf[LocalImageFrame].array(0)
    imageFeature.getHeight() should be (imageFeature.getOriginalHeight)
    imageFeature.getWidth() should be (imageFeature.getOriginalWidth)

    val tmpFile = java.io.File.createTempFile("module", ".jpg")
    Imgcodecs.imwrite(tmpFile.toString, imageFeature.opencvMat())
    println(tmpFile)
  }
}

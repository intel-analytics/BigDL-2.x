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

import java.nio.file.{Files, Paths}

import org.opencv.imgcodecs.Imgcodecs
import org.scalatest.FlatSpec

class CropperSpec extends FlatSpec {
//  "crop" should "work properly" in {
//    val resource = getClass().getClassLoader().getResource("VOCdevkit/VOC2007/JPEGImages/000025.jpg")
//    val img = Files.readAllBytes(Paths.get(resource.getFile))
//
//    val crop = Cropper()
//    val out = crop.cropBytes(img, 0, 0, 100, 200)
//    val mat = MatWrapper.toMat(out)
//    Imgcodecs.imwrite("/tmp/crop.jpg", mat)
//  }
//
//  "crop normalize" should "work properly" in {
//    val resource = getClass().getClassLoader().getResource("VOCdevkit/VOC2007/JPEGImages/000025.jpg")
//    val img = Files.readAllBytes(Paths.get(resource.getFile))
//
//    val crop = Cropper(normalized = true)
//    val out = crop.cropBytes(img, 0, 0, 0.2f, 0.3f)
//    val mat = MatWrapper.toMat(out)
//    Imgcodecs.imwrite("/tmp/crop.jpg", mat)
//  }
}

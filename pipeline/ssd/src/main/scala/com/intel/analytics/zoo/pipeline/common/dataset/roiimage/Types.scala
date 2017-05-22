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

import com.intel.analytics.bigdl.dataset.image.BGRImage
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}

/**
 * Image with region of interest
 * @param imInfo (heightAfterScale, widthAfterScale, scaleh, scalew)
 * @param target image label, optinal
 */
class RoiImage(
  val imInfo: Tensor[Float],
  var target: Target = null) extends BGRImage {

  var path: String = ""

  def this() = {
    this(Tensor[Float](4))
    data = new Array[Float](0)
  }

  // height after scale
  override def height(): Int = imInfo.valueAt(1).toInt

  // width after scale
  override def width(): Int = imInfo.valueAt(2).toInt

  // scale ratio for height
  def scaledH: Float = imInfo.valueAt(3)

  // scale ratio for weidth
  def scaledW: Float = imInfo.valueAt(4)

  def copy(rawData: Array[Byte]): this.type = {
    require(rawData.length == 8 + height * width * 3)
    if (data.length < height * width * 3) {
      data = new Array[Float](height * width * 3)
    }
    var i = 0
    while (i < width * height * 3) {
      data(i) = rawData(i + 8) & 0xff
      i += 1
    }
    this
  }
}

/**
 * Image with byte data and target, path
 * @param data       byte array, this is transformed from java BufferedImage
 * @param dataLength length of byte array
 * @param path       image path
 * @param target     label
 */
case class RoiByteImage(data: Array[Byte], var dataLength: Int, path: String,
  var target: Target = null)


/**
 * Image path and target information
 * @param imagePath image path
 * @param target    image target
 */
case class RoiImagePath(
  imagePath: String,
  target: Target = null) {
}

/**
 * image target with classes and bounding boxes
 * @param classes N (class labels) or 2 * N, the first row is class labels,
 *                the second line is difficults
 * @param bboxes  N * 4
 */
case class Target(classes: Tensor[Float], bboxes: Tensor[Float]) {
  if (classes.dim() == 1) {
    require(classes.size(1) == bboxes.size(1), "the number of classes should be" +
      " equal to the number of bounding box numbers")
  } else if (classes.nElement() > 0 && classes.dim() == 2) {
    require(classes.size(2) == bboxes.size(1), s"the number of classes ${classes.size(2)}" +
      s"should be equal to the number of bounding box numbers ${bboxes.size(1)}")
  }


  def toTable: Table = {
    val table = T()
    table.insert(classes)
    table.insert(bboxes)
  }
}

/**
 * A batch of data feed into the model. The first size is batchsize
 * @param data
 * @param labels
 * @tparam T
 */
case class MiniBatch[T](data: Tensor[T], labels: Tensor[T], imInfo: Tensor[T] = null)

case class SSDByteRecord(data: Array[Byte], path: String)


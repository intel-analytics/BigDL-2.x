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

package com.intel.analytics.bigdl.pipeline.common.dataset.roiimage

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.dataset.image.BGRImage
import com.intel.analytics.bigdl.pipeline.common.BboxUtil
import com.intel.analytics.bigdl.tensor.Tensor

import scala.util.Random

/**
 * resize the image by randomly choosing a scale, resize the target if exists
 * @param scales          array of scale options that for random choice
 * @param scaleMultipleOf Resize test images so that its width and height are multiples of
 * @param resizeRois      whether to resize target bboxes
 * @param maxSize         Max pixel size of the longest side of a scaled input image
 */
class RoiImageResizer(scales: Array[Int], scaleMultipleOf: Int = 1, resizeRois: Boolean = false,
  maxSize: Float = 1000f, isEqualResize: Boolean = false, normalizeRoi: Boolean = false)
  extends Transformer[RoiByteImage, RoiImage] {

  val imageWithRoi = new RoiImage()

  override def apply(prev: Iterator[RoiByteImage]): Iterator[RoiImage] = {
    prev.map(roiByteImage => {
      transform(roiByteImage)
    })
  }

  def transform(roiByteImage: RoiByteImage): RoiImage = {
    // convert byte array back to BufferedImage
    val img = ImageIO.read(new ByteArrayInputStream(roiByteImage.data,
      0, roiByteImage.dataLength))
    val scaleTo = scales(Random.nextInt(scales.length))
    val (height, width) = if (isEqualResize) {
      imageWithRoi.imInfo.setValue(1, scaleTo)
      imageWithRoi.imInfo.setValue(2, scaleTo)
      imageWithRoi.imInfo.setValue(3, img.getHeight().toFloat / scaleTo)
      imageWithRoi.imInfo.setValue(4, img.getWidth().toFloat / scaleTo)
      (scaleTo, scaleTo)
    } else getWidthHeightAfterRatioScale(img, scaleTo, imageWithRoi.imInfo)
    val scaledImage = BGRImage.resizeImage(img, width, height)
    imageWithRoi.copy(scaledImage)
    imageWithRoi.path = roiByteImage.path
    if (resizeRois) {
      require(roiByteImage.target != null, "target is not defined")
      imageWithRoi.target = resizeRois(imageWithRoi.scaledW, imageWithRoi.scaledW,
        roiByteImage.target)
    } else {
      imageWithRoi.target = roiByteImage.target
    }
    if (normalizeRoi) {
      BboxUtil.scaleBBox(imageWithRoi.target.bboxes, 1 / img.getHeight().toFloat,
        1 / img.getWidth().toFloat)
    }
    imageWithRoi
  }

  /**
   * get the width and height of scaled image
   * @param img original image
   * @return imageInfo (scaledHeight, scaledWidth, scaleRatioH, scaleRatioW)
   */
  def getWidthHeightAfterRatioScale(img: BufferedImage, scaleTo: Float,
    imInfo: Tensor[Float]): (Int, Int) = {
    val imSizeMin = Math.min(img.getWidth, img.getHeight)
    val imSizeMax = Math.max(img.getWidth, img.getHeight)
    var im_scale = scaleTo.toFloat / imSizeMin.toFloat
    // Prevent the biggest axis from being more than MAX_SIZE
    if (Math.round(im_scale * imSizeMax) > maxSize) {
      im_scale = maxSize / imSizeMax.toFloat
    }

    val imScaleH = (Math.floor(img.getHeight * im_scale / scaleMultipleOf) *
      scaleMultipleOf / img.getHeight).toFloat
    val imScaleW = (Math.floor(img.getWidth * im_scale / scaleMultipleOf) *
      scaleMultipleOf / img.getWidth).toFloat
    val width = imScaleW * img.getWidth
    val height = imScaleH * img.getHeight
    imInfo.setValue(1, height)
    imInfo.setValue(2, width)
    imInfo.setValue(3, imScaleH)
    imInfo.setValue(4, imScaleW)
    (height.toInt, width.toInt)
  }

  /**
   * resize the ground truth rois
   * @param scaledH
   * @param scaledW
   * @param target
   * @return
   */
  def resizeRois(scaledH: Float, scaledW: Float, target: Target): Target = {
    val num = target.classes.size(2)

    val gtInds = (1 to num).filter(x => target.classes.valueAt(1, x) != 0)
//    val resizedBoxes = Tensor[Float](gtInds.length, 5)
    val resizedBoxes = Tensor[Float](gtInds.length, 4)
    var i = 0
    while (i < gtInds.length) {
      resizedBoxes.setValue(i + 1, 1, target.bboxes.valueAt(gtInds(i), 1) * scaledH)
      resizedBoxes.setValue(i + 1, 2, target.bboxes.valueAt(gtInds(i), 2) * scaledW)
      resizedBoxes.setValue(i + 1, 3, target.bboxes.valueAt(gtInds(i), 3) * scaledH)
      resizedBoxes.setValue(i + 1, 4, target.bboxes.valueAt(gtInds(i), 4) * scaledW)
//      resizedBoxes.setValue(i + 1, 5, target.classes.valueAt(gtInds(i)))
      i += 1
    }
    Target(target.classes, resizedBoxes)
  }
}

object RoiImageResizer {
  def apply(scales: Array[Int], scaleMultipleOf: Int = 1, resizeRois: Boolean = false,
    maxSize: Float = 1000f, isEqualResize: Boolean = false, normalizeRoi: Boolean = false): RoiImageResizer =
    new RoiImageResizer(scales, scaleMultipleOf, resizeRois, maxSize, isEqualResize, normalizeRoi)
}

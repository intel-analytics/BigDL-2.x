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

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.zoo.pipeline.common.BboxUtil
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.transform.vision.label.roi.RoiLabel

/**
 * Image path and target information
 * @param imagePath image path
 * @param target image target
 */
case class RoiImagePath(
  imagePath: String,
  target: RoiLabel = null) {
}

abstract class ImageMiniBatch(val input: Activity, val target: Tensor[Float],
  val imInfo: Tensor[Float] = null) extends MiniBatch[Float] {
  var imageFeatures: Array[ImageFeature] = _
}
/**
 * A batch of data feed into the model. The first size is batchsize
 */
class SSDMiniBatch(feature: Tensor[Float], label: Tensor[Float],
  meta: Tensor[Float] = null) extends ImageMiniBatch(feature, label, meta) {

  private val targetIndices = if (target != null) BboxUtil.getGroundTruthIndices(target) else null

  override def size(): Int = {
    input.toTensor.size(1)
  }

  override def getInput(): Tensor[Float] = {
    input.toTensor
  }

  override def getTarget(): Tensor[Float] = {
    target
  }

  override def slice(offset: Int, length: Int): MiniBatch[Float] = {
    val subInput = input.toTensor.narrow(1, offset, length)
    val subTarget = if (target != null) {
      var i = 0
      val targetOffset = targetIndices(offset - 1)._1
      var targetLength = 0
      while (i < length) {
        targetLength += targetIndices(offset + i - 1)._2
        i += 1
      }
      target.narrow(1, targetOffset, targetLength)
    } else null

    SSDMiniBatch(subInput, subTarget)
  }

  override def set(samples: Seq[Sample[Float]])(implicit ev: TensorNumeric[Float])
  : SSDMiniBatch.this.type = {
    throw new NotImplementedError("do not use Sample here")
  }
}

object SSDMiniBatch {
  def apply(data: Tensor[Float], labels: Tensor[Float], imInfo: Tensor[Float] = null):
  SSDMiniBatch = new SSDMiniBatch(data, labels, imInfo)
}


case class SSDByteRecord(var data: Array[Byte], path: String)


class FrcnnMiniBatch(feature: Table, label: Tensor[Float]) extends ImageMiniBatch(feature, label) {

  private val targetIndices = if (target != null) BboxUtil.getGroundTruthIndices(target) else null

  override def size(): Int = input.toTable.length()

  override def slice(offset: Int, length: Int): MiniBatch[Float] = {
    require(length == 1, "only batch 1 is supported")
    val subInput = input.toTable[Table](offset)
    val subTarget = if (target != null) {
      var i = 0
      val targetOffset = targetIndices(offset - 1)._1
      var targetLength = 0
      while (i < length) {
        targetLength += targetIndices(offset + i - 1)._2
        i += 1
      }
      target.narrow(1, targetOffset, targetLength)
    } else null
    FrcnnMiniBatch(subInput, subTarget)
  }

  override def getInput(): Activity = input

  override def getTarget(): Activity = target

  def getSample(): Table = getInput().toTable[Table](1)

  override def set(samples: Seq[Sample[Float]])(implicit ev: TensorNumeric[Float]): this.type = {
    throw new NotImplementedError("do not use Sample here")
  }
}

object FrcnnMiniBatch {
  def apply(input: Table, target: Tensor[Float]): FrcnnMiniBatch = {
    new FrcnnMiniBatch(input, target)
  }

  def getBboxes(batchInput: Tensor[Float]): Tensor[Float] = {
    batchInput.narrow(2, 4, 4)
  }

  val imageIndex = 1
  val labelIndex = 2
  val difficultIndex = 3
  val x1Index = 4
  val y1Index = 5
  val x2Index = 6
  val y2Index = 7
}


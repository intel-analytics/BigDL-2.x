package com.intel.analytics.zoo.models.image.objectdetection.ssd

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.models.image.objectdetection.common.BboxUtil

/**
  * A batch of data feed into the model. The first size is batchsize
  *
  * @param input
  * @param target
  */
class SSDMiniBatch(val input: Tensor[Float], val target: Tensor[Float],
                   val imInfo: Tensor[Float] = null)
  extends MiniBatch[Float] {

  private val targetIndices = if (target != null) BboxUtil.getGroundTruthIndices(target) else null
  var imageFeatures: Array[ImageFeature] = _

  override def size(): Int = {
    input.size(1)
  }

  override def getInput(): Tensor[Float] = {
    input
  }

  override def getTarget(): Tensor[Float] = {
    target
  }

  override def slice(offset: Int, length: Int): MiniBatch[Float] = {
    val subInput = input.narrow(1, offset, length)
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

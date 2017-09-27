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

package com.intel.analytics.zoo.models.dataset

import com.intel.analytics.bigdl.tensor.Tensor

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

@SerialVersionUID(2531602453913423591L)
class ImageBatch(samples: Seq[ImageSample]) extends Serializable {
  var input = Tensor[Float]()
  val infors = Seq[String]()
  set
  def set(): this.type = {
    require(samples.length > 0, "samples are empty")
    val sizes = Array(samples.length) ++ samples.head.input.size()
    input = Tensor[Float](sizes)
    var offset = 0
    var i = 0
    while (i < samples.length) {
      infors :+ samples(i).infor
      val sampleInput = samples(i).input
      val length = sampleInput.storage().array().length - sampleInput.storageOffset() + 1
      NumericFloat.arraycopy(sampleInput.storage().array(), sampleInput.storageOffset() - 1,
        input.storage().array(),
      input.storageOffset() - 1 + offset, length)
      offset += length
      i += 1
    }
    this
  }
}

object ImageBatch {
  def apply(samples: Seq[ImageSample]): ImageBatch = new ImageBatch(samples)
}

case class ImageSample(val input : Tensor[Float], val infor : String)


// Classification prediction result
case class PredictResult(className : String, credit : Float)

// case class PredictResult(imgInfo : String, array: Array[PredictClass])

object ImageParam extends Enumeration {
  type ImageParam = Value
  val rawImg = Value(0, "RawImg")
  val tensorInput = Value(1, "TensorInput")
  val topN = Value(2, "TopN")
  val paths = Value(3, "Paths")
}
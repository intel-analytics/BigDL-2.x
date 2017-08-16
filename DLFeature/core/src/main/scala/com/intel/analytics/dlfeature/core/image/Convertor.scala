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

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.dlfeature.core.label.{LabelTransformMethod, LabelTransformer}
import com.intel.analytics.dlfeature.core.util.MatWrapper
import org.apache.log4j.Logger

class BytesToFeature(outKey: String = Feature.bytes) extends Transformer[ByteImage, Feature] {
  @transient private var feature: Feature = _

  override def apply(prev: Iterator[ByteImage]): Iterator[Feature] = {
    prev.map(byteImage => {
      if (feature == null) feature = new Feature
      feature(outKey) = byteImage.image
      feature(Feature.path) = byteImage.path
      feature(Feature.label) = byteImage.label
      feature.inKey = outKey
      feature
    })
  }
}

object BytesToFeature {
  def apply(outKey: String = Feature.bytes): BytesToFeature = new BytesToFeature(outKey)
}


class BytesToMat(inKey: String = Feature.bytes, outKey: String = Feature.mat,
  labelTransformMethod: Option[LabelTransformMethod] = None) extends Transformer[Feature, Feature]
  with LabelTransformer {
  if (labelTransformMethod.isDefined) setLabelTransfomer(labelTransformMethod.get)

  override def apply(prev: Iterator[Feature]): Iterator[Feature] = {
    prev.map(feature => {
      val bytes = feature(inKey).asInstanceOf[Array[Byte]]
      var mat: MatWrapper = null
      try {
        mat = MatWrapper.toMat(bytes)
      } catch {
        case e: Exception =>
          e.printStackTrace()
          if (null != mat) mat.release()
          mat = new MatWrapper()
      }
      feature(outKey) = mat
      feature(Feature.width) = mat.width()
      feature(Feature.height) = mat.height()
      feature(Feature.originalW) = mat.width()
      feature(Feature.originalH) = mat.height()
      if (hasLabelTransfomer()) transformLabel(feature)
      feature.inKey = outKey
      feature
    })
  }
}

object BytesToMat {
  def apply(inKey: String = Feature.bytes, outKey: String = Feature.mat,
    labelTransformMethod: Option[LabelTransformMethod] = None): BytesToMat
  = new BytesToMat(inKey, outKey, labelTransformMethod)
}


class MatToFloats(outKey: String = Feature.floats, meanRGB: Option[(Int, Int, Int)] = None)
  extends Transformer[Feature, Feature] {
  @transient private var data: Array[Float] = _
  @transient private var floatMat: MatWrapper = null

  override def apply(prev: Iterator[Feature]): Iterator[Feature] = {
    prev.map(feature => {
      var input: MatWrapper = null
      try {
        input = feature.inputMat()
        val height = input.height()
        val width = input.width()
        if (null == data || data.length < height * width * 3) {
          data = new Array[Float](height * width * 3)
        }
        if (floatMat == null) {
          floatMat = new MatWrapper()
        }
        feature(outKey) = MatWrapper.toFloatBuf(input, data, floatMat)
        if (meanRGB.isDefined) {
          normalize(data, meanRGB.get._1, meanRGB.get._2, meanRGB.get._3)
        }
      } finally {
        feature.inKey = outKey
        if (null != input) input.release()
      }
      feature
    })
  }

  private def normalize(img: Array[Float], meanR: Int, meanG: Int, meanB: Int): Array[Float] = {
    val content = img
    require(content.length % 3 == 0)
    var i = 0
    while (i < content.length) {
      content(i + 2) = content(i + 2) - meanR
      content(i + 1) = content(i + 1) - meanG
      content(i + 0) = content(i + 0) - meanB
      i += 3
    }
    img
  }
}

object MatToFloats {
  val logger = Logger.getLogger(getClass)

  def apply(outKey: String = "floats", meanRGB: Option[(Int, Int, Int)] = None): MatToFloats =
    new MatToFloats(outKey)
}

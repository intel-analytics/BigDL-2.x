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

package com.intel.analytics.zoo.feature.core.image

import com.intel.analytics.zoo.feature.core.util.MatWrapper
import org.apache.log4j.Logger



class BytesToMat()
  extends FeatureTransformer {
  override def transform(feature: Feature): Unit = {
    val bytes = feature(Feature.bytes).asInstanceOf[Array[Byte]]
    var mat: MatWrapper = null
    try {
      mat = MatWrapper.toMat(bytes)
    } catch {
      case e: Exception =>
        e.printStackTrace()
        if (null != mat) mat.release()
        mat = new MatWrapper()
    }
    feature(Feature.mat) = mat
    feature(Feature.originalW) = mat.width()
    feature(Feature.originalH) = mat.height()
  }
}

object BytesToMat {
  def apply(): BytesToMat = new BytesToMat()
}


class MatToFloats(outKey: String = Feature.floats, meanRGB: Option[(Int, Int, Int)] = None)
  extends SingleTransformer[Feature, Feature] {
  @transient private var data: Array[Float] = _
  @transient private var floatMat: MatWrapper = null

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

  override def apply(feature: Feature): Feature = {
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
      feature(Feature.width) = input.width()
      feature(Feature.height) = input.height()
    } finally {
      if (null != input) input.release()
    }
    feature
  }
}

object MatToFloats {
  val logger = Logger.getLogger(getClass)

  def apply(outKey: String = "floats", meanRGB: Option[(Int, Int, Int)] = None): MatToFloats =
    new MatToFloats(outKey)
}

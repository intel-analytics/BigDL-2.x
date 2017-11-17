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

package com.intel.analytics.zoo.pipeline.ssd.model

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import SSD._
import com.intel.analytics.zoo.pipeline.common.nn.PostProcessParam

object SSDAlexNet {

  def alexnetPart1(): Sequential[Float] = {
    val model = Sequential()
    addConvRelu(model, (3, 96, 11, 4, 5), "1")
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("norm1"))
    model.add(SpatialMaxPooling(3, 3, 2, 2, 1, 1).setName("pool1"))

    addConvRelu(model, (96, 256, 5, 1, 2), "2", nGroup = 2)
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("norm2"))

    addConvRelu(model, (256, 384, 3, 1, 1), "3")

    addConvRelu(model, (384, 384, 3, 1, 1), "4", nGroup = 2)
    model
  }

  def alexnetPart2(): Sequential[Float] = {
    val model = Sequential()
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2"))

    addConvRelu(model, (384, 256, 3, 1, 1), "5", nGroup = 2)
    model.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).setName("pool5"))
  }

  def apply(numClasses: Int, resolution: Int = 300,
    postProcessParam: Option[PostProcessParam] = None): Module[Float] = {
    if (resolution == 300) {
      val isClip = true
      val isFlip = true
      val variances = Array(0.1f, 0.1f, 0.2f, 0.2f)
      val aspectRatios = Array(2f, 3f)
      val step = 0
      var params = Map[String, ComponetParam]()
      params += "conv4_3_norm" -> ComponetParam(384, 3, minSizes = Array(30f), null,
        aspectRatios = Array(2), isFlip, isClip, variances, step)
      params += "fc7" -> ComponetParam(256, 6, minSizes = Array(60f), maxSizes = Array(114f),
        aspectRatios, isFlip, isClip, variances, step)
      params += "conv6_2" -> ComponetParam(512, 6, minSizes = Array(114f),
        maxSizes = Array(168f), aspectRatios, isFlip, isClip, variances, step)
      params += "conv7_2" -> ComponetParam(256, 6, minSizes = Array(168f),
        maxSizes = Array(222f), aspectRatios, isFlip, isClip, variances, step)
      params += "conv8_2" -> ComponetParam(256, 6, minSizes = Array(222f),
        maxSizes = Array(276f), aspectRatios, isFlip, isClip, variances, step)
      params += "pool6" -> ComponetParam(256, 6, minSizes = Array(276f),
        maxSizes = Array(330f), aspectRatios, isFlip, isClip, variances, step)

      val postParam = postProcessParam.getOrElse(PostProcessParam(numClasses))
      val model = SSD(numClasses, resolution = 300, alexnetPart1(), alexnetPart2(),
        params, isLastPool = true, normScale = 19.9096f, postParam)
      val namedModules = Utils.getNamedModules(model)
      namedModules("fc6").setName("fc6-conv")
      namedModules("fc7").setName("fc7-conv")
      model
    } else {
      throw new Exception("currently only support 300 * 300")
    }
  }
}

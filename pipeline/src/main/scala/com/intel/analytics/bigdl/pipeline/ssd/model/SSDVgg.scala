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

package com.intel.analytics.bigdl.pipeline.ssd.model

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.pipeline.ssd.PostProcessParam
import com.intel.analytics.bigdl.pipeline.ssd.model.SSD._

object SSDVgg {

  private def vgg16Part1(): Sequential[Float] = {
    val vggNetPart1 = Sequential()
    addConvRelu(vggNetPart1, (3, 64, 3, 1, 1), "1_1")
    addConvRelu(vggNetPart1, (64, 64, 3, 1, 1), "1_2")
    vggNetPart1.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool1"))

    addConvRelu(vggNetPart1, (64, 128, 3, 1, 1), "2_1")
    addConvRelu(vggNetPart1, (128, 128, 3, 1, 1), "2_2")
    vggNetPart1.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool2"))

    addConvRelu(vggNetPart1, (128, 256, 3, 1, 1), "3_1")
    addConvRelu(vggNetPart1, (256, 256, 3, 1, 1), "3_2")
    addConvRelu(vggNetPart1, (256, 256, 3, 1, 1), "3_3")
    vggNetPart1.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool3"))

    addConvRelu(vggNetPart1, (256, 512, 3, 1, 1), "4_1")
    addConvRelu(vggNetPart1, (512, 512, 3, 1, 1), "4_2")
    addConvRelu(vggNetPart1, (512, 512, 3, 1, 1), "4_3")
    vggNetPart1
  }

  private def vgg16Part2(): Sequential[Float] = {
    val vggNetPart2 = Sequential()
    vggNetPart2.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool4"))
    addConvRelu(vggNetPart2, (512, 512, 3, 1, 1), "5_1")
    addConvRelu(vggNetPart2, (512, 512, 3, 1, 1), "5_2")
    addConvRelu(vggNetPart2, (512, 512, 3, 1, 1), "5_3")
    vggNetPart2.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName("pool5"))
    vggNetPart2
  }

  def apply(numClasses: Int, resolution: Int = 300, dataset: String = "coco",
    sizes: Option[Array[Float]] = None): Module[Float] = {
    require(resolution == 300, "only support 300*300 as input")
    require(dataset == "coco", "currently only support coco model")
    val isClip = false
    val isFlip = true
    val variances = Array(0.1f, 0.1f, 0.2f, 0.2f)
    var params = Map[String, ComponetParam]()
    val priorBoxSizes = Array[Float](21, 45, 99, 153, 207, 261, 315)

    if (resolution == 300) {
      require(priorBoxSizes.length == 7, "the min and max division boundary length should be 7")
      params += "conv4_3_norm" -> ComponetParam(512, 4,
        minSizes = Array(priorBoxSizes(0)), maxSizes = Array(priorBoxSizes(1)),
        aspectRatios = Array(2), isFlip, isClip, variances, 8)
      params += "fc7" -> ComponetParam(512, 6,
        minSizes = Array(priorBoxSizes(1)), maxSizes = Array(priorBoxSizes(2)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 16)
      params += "conv6_2" -> ComponetParam(512, 6,
        minSizes = Array(priorBoxSizes(2)), maxSizes = Array(priorBoxSizes(3)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 32)
      params += "conv7_2" -> ComponetParam(256, 6,
        minSizes = Array(priorBoxSizes(3)), maxSizes = Array(priorBoxSizes(4)),
        aspectRatios = Array(2, 3), isFlip, isClip, variances, 64)
      params += "conv8_2" -> ComponetParam(256, 4,
        minSizes = Array(priorBoxSizes(4)), maxSizes = Array(priorBoxSizes(5)),
        aspectRatios = Array(2), isFlip, isClip, variances, 100)
      params += "conv9_2" -> ComponetParam(256, 4,
        minSizes = Array(priorBoxSizes(5)), maxSizes = Array(priorBoxSizes(6)),
        aspectRatios = Array(2), isFlip, isClip, variances, 300)
      SSD(numClasses, resolution, vgg16Part1(), vgg16Part2(), params, normScale = 13.25217724f,
        isLastPool = false, param = PostProcessParam(numClasses))
    } else throw new NotImplementedError("currently only resolution 300x300 is supported")
  }
}

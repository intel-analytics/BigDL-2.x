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

package com.intel.analytics.zoo.pipeline.fasterrcnn.model

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.FasterRcnn._

object VggFRcnnSeq {

  def vgg16(): Sequential[Float] = {
    val vggNet = Sequential()
    def addConvRelu(param: (Int, Int, Int, Int, Int), name: String)
    : Unit = {
      vggNet.add(conv(param, s"conv$name"))
      vggNet.add(ReLU(true).setName(s"relu$name"))
    }
    addConvRelu((3, 64, 3, 1, 1), "1_1")
    addConvRelu((64, 64, 3, 1, 1), "1_2")
    vggNet.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool1"))

    addConvRelu((64, 128, 3, 1, 1), "2_1")
    addConvRelu((128, 128, 3, 1, 1), "2_2")
    vggNet.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool2"))

    addConvRelu((128, 256, 3, 1, 1), "3_1")
    addConvRelu((256, 256, 3, 1, 1), "3_2")
    addConvRelu((256, 256, 3, 1, 1), "3_3")
    vggNet.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool3"))

    addConvRelu((256, 512, 3, 1, 1), "4_1")
    addConvRelu((512, 512, 3, 1, 1), "4_2")
    addConvRelu((512, 512, 3, 1, 1), "4_3")
    vggNet.add(SpatialMaxPooling(2, 2, 2, 2).ceil().setName("pool4"))

    addConvRelu((512, 512, 3, 1, 1), "5_1")
    addConvRelu((512, 512, 3, 1, 1), "5_2")
    addConvRelu((512, 512, 3, 1, 1), "5_3")
    vggNet
  }

  private def rpn(anchorNum: Int): Sequential[Float] = {
    val rpnModel = Sequential()
    rpnModel.add(conv((512, 512, 3, 1, 1), "rpn_conv/3x3"))
    rpnModel.add(ReLU(true).setName("rpn_relu/3x3"))
    val clsAndReg = ConcatTable()
    val clsSeq = Sequential()
    clsSeq.add(conv((512, 18, 1, 1, 0), "rpn_cls_score"))
    clsSeq.add(InferReshape(Array(0, 2, -1, 0)))
    clsSeq.add(SoftMax()).add(InferReshape(Array(1, 2 * anchorNum, -1, 0)))
    clsAndReg.add(clsSeq)
      .add(conv((512, 36, 1, 1, 0), "rpn_bbox_pred"))
    rpnModel.add(clsAndReg)
    rpnModel
  }

  def baseAndRpn(anchorNum: Int): Sequential[Float] = {
    val compose = Sequential()
    compose.add(vgg16())
    val vggRpnModel = ConcatTable()
    vggRpnModel.add(rpn(anchorNum))
    vggRpnModel.add(Identity())
    compose.add(vggRpnModel)
    compose
  }

  def fastRcnn(): Sequential[Float] = {
    val pool = 7
    val model = Sequential()
      .add(RoiPooling(pool, pool, 0.0625f).setName("pool5"))
      .add(InferReshape(Array(-1, 512 * pool * pool)))
      .add(Linear(512 * pool * pool, 4096).setName("fc6"))
      .add(ReLU())
      .add(Dropout().setName("drop6"))
      .add(Linear(4096, 4096).setName("fc7"))
      .add(ReLU())
      .add(Dropout().setName("drop7"))

    val cls = Sequential().add(Linear(4096, 21).setName("cls_score"))
    cls.add(SoftMax().setName("cls_prob"))
    val clsReg = ConcatTable()
      .add(cls)
      .add(Linear(4096, 84).setName("bbox_pred"))

    model.add(clsReg)
    model
  }


  @deprecated
  def apply(nClass: Int): Module[Float] = {
    val rpnPreNmsTopN = 6000
    val rpnPostNmsTopN = 300
    val ratios = VggFRcnn.ratios
    val scales = VggFRcnn.scales
    val anchorNum = scales.length * ratios.length
    FasterRcnn(nClass, rpnPreNmsTopN,
      rpnPostNmsTopN, ratios, scales, baseAndRpn(anchorNum), fastRcnn())
  }
}


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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.zoo.pipeline.fasterrcnn.model.FasterRcnn._


object PvanetFRcnn {

  private def concatNeg(name: String): Concat[Float] = {
    val concat = Concat(2)
    concat.add(Identity())
    concat.add(Power(1, -1, 0).setName(s"$name/neg"))
    concat.setName(s"$name/concat")
    concat
  }


  private def addScaleRelu(module: Sequential[Float],
    sizes: Array[Int], name: String): Unit = {
    module.add(Scale(sizes).setName(name))
    module.add(ReLU())
  }

  private def addConvComponent(model: Sequential[Float],
    compId: Int, index: Int, p: Array[(Int, Int, Int, Int, Int)]) = {
    val label = s"${compId}_$index"
    val convTable = ConcatTable()
    val conv_left = Sequential()
    var i = 0
    if (index == 1) {
      conv_left.add(conv(p(i), s"conv$label/1/conv"))
      i += 1
    } else {
      conv_left.add(conv(p(i), s"conv$label/1/conv"))
      i += 1
    }

    conv_left.add(SpatialBatchNormalization(p(i)._2, 1e-3).setName(s"conv$label/2/bn"))
    conv_left.add(Scale(Array(1, p(i)._2, 1, 1)).setName(s"conv$label/2/bn_scale"))
    conv_left.add(ReLU().setName(s"conv$label/2/relu"))
    conv_left.add(conv(p(i), s"conv$label/2/conv"))
    i += 1
    conv_left.add(concatNeg(s"conv$label/2"))
    if (compId == 2) {
      addScaleRelu(conv_left, Array(1, 48, 1, 1), s"conv$label/2/bn_scale")
    } else {
      addScaleRelu(conv_left, Array(1, 96, 1, 1), s"conv$label/2/bn_scale")
    }

    conv_left.add(conv(p(i), s"conv$label/3/conv"))
    i += 1

    convTable.add(conv_left)
    if (index == 1) {
      convTable.add(conv(p(i), s"conv$label/proj"))
      i += 1
    } else {
      convTable.add(Identity())
    }
    model.add(convTable)
    model.add(CAddTable().setName(s"conv$label"))
  }

  private def addInception(module: Sequential[Float], label: String, index: Int,
    p: Array[(Int, Int, Int, Int, Int)]): Unit = {
    val left = Sequential()
    val incep = Concat(2)

    var i = 0
    val com1 = Sequential()
    com1.add(conv(p(i), s"conv$label/incep/0/conv", false)).add(ReLU())
    i += 1
    incep.add(com1)

    val com2 = Sequential()
    com2.add(conv(p(i), s"conv$label/incep/1_reduce/conv", false)).add(ReLU())
    i += 1
    com2.add(conv(p(i), s"conv$label/incep/1_0/conv", false)).add(ReLU())
    i += 1
    incep.add(com2)

    val com3 = Sequential()
    com3.add(conv(p(i), s"conv$label/incep/2_reduce/conv", false)).add(ReLU())
    i += 1
    com3.add(conv(p(i), s"conv$label/incep/2_0/conv", false)).add(ReLU())
    i += 1
    com3.add(conv(p(i), s"conv$label/incep/2_1/conv", false)).add(ReLU())
    i += 1
    incep.add(com3)

    if (index == 1) {
      val com4 = Sequential()
      com4.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName(s"conv$label/incep/pool"))
      com4.add(conv(p(i), s"conv$label/incep/poolproj/conv", false)).add(ReLU())
      i += 1
      incep.add(com4)
    }

    left.add(incep)
    left.add(conv(p(i), s"conv$label/out/conv"))
    i += 1
    val table = ConcatTable()
    table.add(left)
    if (index == 1) {
      table.add(conv(p(i), s"conv$label/proj"))
      i += 1
    } else {
      table.add(Identity())
    }
    module.add(table)
    module.add(CAddTable().setName(s"conv$label"))
  }


  private def pvanet: Sequential[Float] = {
    val model = Sequential()
    model.add(conv((3, 16, 7, 2, 3), "conv1_1/conv", false))

    model.add(concatNeg("conv1_1"))
    addScaleRelu(model, Array(1, 32, 1, 1), "conv1_1/scale")
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1"))


    addConvComponent(model, 2, 1, Array((32, 24, 1, 1, 0), (24, 24, 3, 1, 1),
      (48, 64, 1, 1, 0), (32, 64, 1, 1, 0)))
    var i = 2
    while (i <= 3) {
      addConvComponent(model, 2, i, Array((64, 24, 1, 1, 0), (24, 24, 3, 1, 1), (48, 64, 1, 1, 0)))
      i += 1
    }

    addConvComponent(model, 3, 1, Array((64, 48, 1, 2, 0), (48, 48, 3, 1, 1),
      (96, 128, 1, 1, 0), (64, 128, 1, 2, 0)))

    i = 2
    while (i <= 4) {
      addConvComponent(model, 3, i, Array((128, 48, 1, 1, 0),
        (48, 48, 3, 1, 1), (96, 128, 1, 1, 0)))
      i += 1
    }

    val inceptions4_5 = Sequential()

    val inceptions4 = Sequential()
    addInception(inceptions4, "4_1", 1, Array((128, 64, 1, 2, 0), (128, 48, 1, 2, 0),
      (48, 128, 3, 1, 1), (128, 24, 1, 2, 0), (24, 48, 3, 1, 1), (48, 48, 3, 1, 1),
      (128, 128, 1, 1, 0), (368, 256, 1, 1, 0), (128, 256, 1, 2, 0)))

    i = 2
    while (i <= 4) {
      addInception(inceptions4, s"4_$i", i, Array((256, 64, 1, 1, 0), (256, 64, 1, 1, 0),
        (64, 128, 3, 1, 1), (256, 24, 1, 1, 0), (24, 48, 3, 1, 1),
        (48, 48, 3, 1, 1), (240, 256, 1, 1, 0)))
      i += 1
    }
    inceptions4_5.add(inceptions4)


    val seq5 = Sequential()
    val inceptions5 = Sequential()
    addInception(inceptions5, "5_1", 1, Array((256, 64, 1, 2, 0), (256, 96, 1, 2, 0),
      (96, 192, 3, 1, 1), (256, 32, 1, 2, 0), (32, 64, 3, 1, 1), (64, 64, 3, 1, 1),
      (256, 128, 1, 1, 0), (448, 384, 1, 1, 0), (256, 384, 1, 2, 0)))
    i = 2
    while (i <= 4) {
      addInception(inceptions5, s"5_$i", i, Array((384, 64, 1, 1, 0), (384, 96, 1, 1, 0),
        (96, 192, 3, 1, 1), (384, 32, 1, 1, 0), (32, 64, 3, 1, 1), (64, 64, 3, 1, 1),
        (320, 384, 1, 1, 0)))
      i += 1
    }

    seq5.add(inceptions5)
    seq5.add(SpatialFullConvolution(384, 384, 4, 4, 2, 2, 1, 1,
      nGroup = 384, noBias = true).setName("upsample"))

    val concat5 = Concat(2)
    concat5.add(Identity())
    concat5.add(seq5)

    inceptions4_5.add(concat5)

    val concatConvf = Concat(2).setName("concat")
    concatConvf.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("downsample"))
    concatConvf.add(inceptions4_5)
    model.add(concatConvf)

    model
  }

  def baseAndRpn(anchorNum: Int): Sequential[Float] = {
    val compose = Sequential()
    compose.add(pvanet)

    val convTable = new ConcatTable[Float]
    convTable.add(Sequential()
      .add(conv((768, 128, 1, 1, 0), "convf_rpn"))
      .add(ReLU()))
    convTable.add(Sequential()
      .add(conv((768, 384, 1, 1, 0), "convf_2"))
      .add(ReLU()))
    compose.add(convTable)
    val rpnAndFeature = ConcatTable()
    rpnAndFeature.add(Sequential()
      .add(new SelectTable(1)).add(rpn(anchorNum)))
    rpnAndFeature.add(JoinTable(2, 4))
    compose.add(rpnAndFeature)
    compose
  }

  private def fastRcnn(): Sequential[Float] = {
    val pool = 6
    val model = Sequential()
      .add(RoiPooling(pool, pool, 0.0625f).setName("roi_pool_conv5"))
      .add(InferReshape(Array(-1, 512 * pool * pool)).setName("roi_pool_conv5_reshape"))
    model.add(Linear(512 * pool * pool, 4096).setName("fc6"))
    model.add(ReLU())
    model.add(Linear(4096, 4096).setName("fc7"))
    model.add(ReLU())

    val cls = Sequential().add(Linear(4096, 21).setName("cls_score"))
    cls.add(SoftMax())
    val clsReg = ConcatTable()
      .add(cls)
      .add(Linear(4096, 84).setName("bbox_pred"))

    model.add(clsReg)
    model
  }

  private def rpn(anchorNum: Int): Sequential[Float] = {
    val rpnModel = Sequential()
    rpnModel.add(conv((128, 384, 3, 1, 1), "rpn_conv1"))
    rpnModel.add(ReLU())
    val clsAndReg = ConcatTable()
    val clsSeq = Sequential()
    clsSeq.add(conv((384, 50, 1, 1, 0), "rpn_cls_score"))
    clsSeq.add(InferReshape(Array(0, 2, -1, 0)).setName("rpn_cls_score_reshape"))
    clsSeq.add(SoftMax())
      .add(InferReshape(Array(1, 2 * anchorNum, -1, 0))
        .setName("rpn_cls_prob_reshape"))
    clsAndReg.add(clsSeq).add(conv((384, 100, 1, 1, 0), "rpn_bbox_pred"))
    rpnModel.add(clsAndReg)
    rpnModel
  }


  @deprecated
  def apply(nClass: Int, postProcessParam: PostProcessParam): Module[Float] = {
    val scales = Array[Float](3, 6, 9, 16, 32)
    val ratios = Array(0.5f, 0.667f, 1.0f, 1.5f, 2.0f)
    // Number of top scoring boxes to keep before apply NMS to RPN proposals
    val rpnPreNmsTopN = 12000
    // Number of top scoring boxes to keep after applying NMS to RPN proposals
    val rpnPostNmsTopN = 200
    FasterRcnn(nClass, rpnPreNmsTopN, rpnPostNmsTopN, ratios, scales,
      baseAndRpn(ratios.length * scales.length), fastRcnn())
  }
}

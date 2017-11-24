/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.fasterrcnn.model

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.zoo.pipeline.common.nn.{FrcnnCriterion, ProposalTarget}
import com.intel.analytics.zoo.pipeline.fasterrcnn.TestUtil
import org.scalatest.{FlatSpec, Matchers}

class VggFRcnnSpec extends FlatSpec with Matchers {
  "faster rcnn graph" should "forward properly" in {
    val frcnnGraph = VggFRcnn(21,
      PostProcessParam(0.3f, 21, false, -1, 0)).evaluate()
    val frcnn = VggFRcnnSeq(21).evaluate()
    frcnnGraph.loadModelWeights(frcnn)

    val data = Tensor[Float](1, 3, 300, 300).randn()
    val rois = Tensor[Float](T(300f, 300f, 1f, 1f)).resize(1, 4)
    val input = T(data, rois, null)
    val input2 = T(data, rois)
    frcnnGraph.forward(input)
    frcnn.forward(input2)

//    frcnnGraph.output.toTable.length() should equal(frcnn.output.toTable.length())

    val namedModule = Utils.getNamedModules(frcnnGraph)
    val namedModule2 = Utils.getNamedModules(frcnn)
    namedModule.keys.foreach(key => {
      if (namedModule.contains(key) && namedModule2.contains(key)) {
        val out1 = namedModule(key).output
        val out2 = namedModule2(key).output
        if (out1.isTensor) {
          if (pass(out1.toTensor, out2.toTensor)) {
//            println(s"${key} pass")
          } else {
            println(s"${key} not pass")
          }
        }
      }
    })
  }

  def pass(out1: Tensor[Float], out2: Tensor[Float]): Boolean = {
    if (out1.nElement() != out2.nElement()) return false
    var status = true
    out1.toTensor[Float].map(out2.toTensor[Float], (a, b) => {
      if (Math.abs(a - b) > 1e-6) {
//        println(a, b)
        status = false
      }
      a
    })
    status
  }

  "save module" should "work" in {
    VggFRcnn.debug = true
    val module = VggFRcnn(21,
      PostProcessParam(0.3f, 21, false, -1, 0))
    module.evaluate()
    val target = Tensor(Storage(Array(0.0, 11.0, 0.0, 0.337411, 0.468211, 0.429096, 0.516061)
      .map(_.toFloat))).resize(1, 7)
    val data = Tensor[Float](1, 3, 200, 300).randn()
    val imInfo = Tensor[Float](T(200, 300, 1, 1)).resize(1, 4)
    val input = T(data.clone(), imInfo.clone(), target.clone())
    val res1 = module.forward(input).toTensor[Float].clone()

    val tmpFile = java.io.File.createTempFile("module", ".bigdl")
    module.saveModule(tmpFile.getAbsolutePath, true)
    val loaded = Module.loadModule[Float](tmpFile.getAbsolutePath).evaluate()
    val input2 = T(data.clone(), imInfo.clone(), target.clone())
    val res2 = loaded.forward(input2).toTensor[Float]
    res1 should be(res2)
    if (tmpFile.exists()) {
      tmpFile.delete()
    }
  }

  "forward backward" should "work" in {
    val target = Tensor(Storage(Array(0.0, 11.0, 0.0, 0.337411, 0.468211, 0.429096, 0.516061)
      .map(_.toFloat))).resize(1, 7)
    val frcnn = VggFRcnn(21, PostProcessParam(0.3f, 21, false, 100, 0.05))
    val criterion = FrcnnCriterion()
    val input = T()
    input.insert(Tensor[Float](1, 3, 600, 800).randn())
    input.insert(Tensor[Float](T(600, 800, 1, 1)).resize(1, 4))
    input.insert(target)

    frcnn.forward(input)
    criterion.forward(frcnn.output.toTable, target)
    criterion.backward(frcnn.output.toTable, target)

    frcnn.backward(input, criterion.gradInput)

  }

  "forward backward with norm" should "work" in {
    val target = Tensor(Storage(Array(0.0, 11.0, 0.0, 0.337411, 0.468211, 0.429096, 0.516061)
      .map(_.toFloat))).resize(1, 7)
    val model = VggFRcnn(21, PostProcessParam(0.3f, 21, false, 100, 0.05)).evaluate()
    model.saveModule("/tmp/tmp.model", true)
    val criterion = FrcnnCriterion()
    val input = T()
    input.insert(Tensor[Float](1, 3, 600, 800).randn())
    input.insert(Tensor[Float](T(600, 800, 1, 1)).resize(1, 4))
    input.insert(target)
    val bboxPred = model("bbox_pred").get

    model.forward(input)

    val means = ProposalTarget.BBOX_NORMALIZE_MEANS.reshape(Array(1, 4))
      .expand(Array(21, 4)).reshape(Array(21 * 4))
    val stds = ProposalTarget.BBOX_NORMALIZE_STDS.reshape(Array(1, 4))
      .expand(Array(21, 4)).reshape(Array(21 * 4))
    val bp1 = bboxPred.output.toTensor.clone()
    bp1.cmul(stds.reshape(Array(1, 21 * 4)).expand(Array(bp1.size(1), 21 * 4)))
    println(bp1)

    val wbs = bboxPred.getWeightsBias()
    wbs(0).cmul(stds.reshape(Array(21 * 4, 1)).expand(Array(21 * 4, wbs(0).size(2))))
    wbs(1).cmul(stds).add(1, means)
    model.forward(input)
    val bp2 = bboxPred.output.toTensor.clone()
    println(bp2)

    bp1 should be (bp2)
  }


  "forward with maxPerImage" should "work" in {
    val target = Tensor(Storage(Array(0.0, 11.0, 0.0, 0.337411, 0.468211, 0.429096, 0.516061)
      .map(_.toFloat))).resize(1, 7)
    val frcnn = VggFRcnn(21, PostProcessParam(0.3f, 21, false, 100, 0.05)).evaluate()
    val input = T()
    input.insert(Tensor[Float](1, 3, 600, 800).randn())
    input.insert(Tensor[Float](T(600, 800, 1, 1)).resize(1, 4))
    input.insert(target)

    frcnn.forward(input)
  }

  "zero tensor resize" should "work" in {
    val t1 = Tensor[Float](10)
    val t2 = Tensor[Float](0)
    t1.set()
    t2.size() should be(Array(0))
    t1.nElement() should be(t2.nElement())
  }

  "forward backward" should "work properly" in {
    TestUtil.middleRoot = "/home/jxy/data/middle/vgg16/new"
    val target = Tensor(Storage(Array(
      0, 14, 0,
      2.702702636718750000e+02,
      1.573573608398437500e+02,
      3.495495605468750000e+02,
      2.654654541015625000e+02,
      0, 15, 0,
      2.726726684570312500e+02,
      1.273273239135742188e+02,
      3.375375366210937500e+02,
      2.234234161376953125e+02,
      0, 15, 1,
      5.285285186767578125e+01,
      3.339339294433593750e+02,
      7.687687683105468750e+01,
      3.915915832519531250e+02)
      .map(_.toFloat))).resize(3, 7)
    val frcnn = Module.loadCaffe(VggFRcnn(21,
      PostProcessParam(0.3f, 21, false, -1, 0)),
      "/home/jxy/data/caffeModels/vgg16/test.prototxt",
      "/home/jxy/data/middle/vgg16/new/pretrained.caffemodel", false)
    val criterion = FrcnnCriterion()
    val (weights, grads) = frcnn.getParameters()
    val state = T(
      "learningRate" -> 0.001,
      "weightDecay" -> 0.0005,
      "momentum" -> 0.9,
      "dampening" -> 0.0,
      "learningRateSchedule" -> SGD.MultiStep(Array(80000, 100000, 120000), 0.1)
    )
    val sgd = new SGD[Float]

    frcnn.zeroGradParameters()
    val input = T()
    input.insert(TestUtil.loadFeatures("data"))
    input.insert(Tensor[Float](T(400, 601, 1.2012012, 1.2012012)).resize(1, 4))
    input.insert(target.clone())

    frcnn.zeroGradParameters()
    frcnn.forward(input)
    criterion.forward(frcnn.output.toTable, target)
    criterion.backward(frcnn.output.toTable, target)
    frcnn.backward(input, criterion.gradInput)
    println(s"loss: ${criterion.output}")
//    val bboxPredDiff = TestUtil.loadFeatures("bbox_preddiff")
//    println(Tensor.sparse(bboxPredDiff))
//    println("bigdl =========== bbox pred grad")
//    println(Tensor.sparse(criterion.gradInput.toTable(2)))
//    TestUtil.assertEqual2(TestUtil.loadFeatures("conv5_3"),
//      frcnn("relu5_3").get.output.toTensor[Float], "conv5_3", 1e-4)
//    TestUtil.assertEqual("rpn_bbox_pred", frcnn("rpn_bbox_pred").get.output.toTensor[Float], 1e-4)
//    TestUtil.assertEqual2(TestUtil.loadFeatures("rpn_rois"),
//      frcnn("proposal").get.output.toTensor[Float], "rpn_rois", 1e-3)
//    TestUtil.assertEqual2(TestUtil.loadFeatures("rois"),
//      frcnn("roi-data").get.output.toTable[Tensor[Float]](1), "rois", 1e-3)
//    TestUtil.assertEqual2(TestUtil.loadFeatures("bbox_targets"),
//      frcnn("roi-data").get.output.toTable[Tensor[Float]](3), "bbox_targets", 1e-3)

    sgd.optimize(_ => (criterion.output, grads), weights, state, state)

    frcnn.zeroGradParameters()
    frcnn.forward(input)
    criterion.forward(frcnn.output.toTable, target)
    criterion.backward(frcnn.output.toTable, target)
    frcnn.backward(input, criterion.gradInput)
    println(s"loss: ${criterion.output}")

    frcnn.evaluate()
    val input2 = T()
    input2.insert(TestUtil.loadFeatures("data"))
    input2.insert(Tensor[Float](T(400, 601, 1.2012012, 1.2012012)).resize(1, 4))
    input2.insert(Tensor[Float]())
    frcnn.forward(input2)
  }

  "time before compress" should "work" in {
    val model = Module.loadModule[Float]("/home/jxy/code/analytics-zoo/pipeline/ssd/data/models/" +
      "bigdl_frcnn_vgg_voc.model")

    val data = Tensor[Float](1, 3, 600, 800).randn()
    val rois = Tensor[Float](T(600f, 800f, 1f, 1f)).resize(1, 4)
    val input = T(data, rois, null)

    model.evaluate()
    model.forward(input)
    model.resetTimes()


    val start = System.nanoTime()
    model.forward(input)
    println("time takes: ", (System.nanoTime() - start) / 1e9 + "s")

    val namedModules = Utils.getNamedModules(model)

    var convTime: Double = 0
    var reluTime: Double = 0
    var other: Double = 0
    namedModules.foreach(x => {
      if (x._2.isInstanceOf[SpatialConvolution[Float]]) {
        convTime += x._2.getTimes()(0)._2 / 1e9
      } else if (x._2.isInstanceOf[ReLU[Float]]) {
        reluTime += x._2.getTimes()(0)._2 / 1e9
      } else if (x._2.isInstanceOf[Linear[Float]]) {
        println(x._1 + "\t" + x._2.getTimes()(0)._2 / 1e9)
      }
      else if (x._2.isInstanceOf[RoiPooling[Float]]) {
        println(x._1 + "\t" + x._2.getTimes()(0)._2 / 1e9)
      }
      else {
        other += x._2.getTimes()(0)._2 / 1e9
      }
    })
    println(s"convolution\t$convTime")
    println(s"relu\t$reluTime")
    println(s"other\t$other")
  }

  "time after compress" should "work" in {

    val model = Module.loadModule[Float]("/home/jxy/code/analytics-zoo/pipeline/ssd/data/models/" +
      "bigdl_frcnn_vgg_voc_compress.model")

    val data = Tensor[Float](1, 3, 600, 800).randn()
    val rois = Tensor[Float](T(600f, 800f, 1f, 1f)).resize(1, 4)
    val input = T(data, rois, null)

    model.evaluate()
    model.forward(input)
    model.resetTimes()


    val start = System.nanoTime()
    model.forward(input)
    println("time takes: ", (System.nanoTime() - start) / 1e9)

    val namedModules = Utils.getNamedModules(model)

    var convTime: Double = 0
    var reluTime: Double = 0
    var other: Double = 0
    namedModules.foreach(x => {
      if (x._2.isInstanceOf[SpatialConvolution[Float]]) {
        convTime += x._2.getTimes()(0)._2 / 1e9
      } else if (x._2.isInstanceOf[ReLU[Float]]) {
        reluTime += x._2.getTimes()(0)._2 / 1e9
      } else if (x._2.isInstanceOf[Linear[Float]]) {
        println(x._1 + "\t" + x._2.getTimes()(0)._2 / 1e9)
      }
      else if (x._2.isInstanceOf[RoiPooling[Float]]) {
        println(x._1 + "\t" + x._2.getTimes()(0)._2 / 1e9)
      }
      else {
        other += x._2.getTimes()(0)._2 / 1e9
      }
    })
    println(s"convolution\t$convTime")
    println(s"relu\t$reluTime")
    println(s"other\t$other")
  }

  "time for training" should "work" in {
    val target = Tensor(Storage(Array(0.0, 11.0, 0.0, 0.337411, 0.468211, 0.429096, 0.516061)
      .map(_.toFloat))).resize(1, 7)
    val model = VggFRcnn(21, PostProcessParam(0.3f, 21, false, 100, 0.05))
    val criterion = FrcnnCriterion()
    val input = T()
    input.insert(Tensor[Float](1, 3, 600, 800).randn())
    input.insert(Tensor[Float](T(600, 800, 1, 1)).resize(1, 4))
    input.insert(target)

    model.forward(input)
    criterion.forward(model.output.toTable, target)
    criterion.backward(model.output.toTable, target)

    model.backward(input, criterion.gradInput)

    model.resetTimes()


    val start = System.nanoTime()
    model.forward(input)
    criterion.forward(model.output.toTable, target)
    criterion.backward(model.output.toTable, target)

    model.backward(input, criterion.gradInput)
    println("time takes: ", (System.nanoTime() - start) / 1e9)

    val namedModules = Utils.getNamedModules(model)

    var convTime: Double = 0
    var reluTime: Double = 0
    var other: Double = 0
    namedModules.foreach(x => {
      if (x._2.isInstanceOf[SpatialConvolution[Float]]) {
        convTime += (x._2.getTimes()(0)._2 / 1e9 + x._2.getTimes()(0)._3 / 1e9)
      } else if (x._2.isInstanceOf[ReLU[Float]]) {
        reluTime += (x._2.getTimes()(0)._2 / 1e9 + x._2.getTimes()(0)._3 / 1e9)
      } else {
        println(x._1 + "\t" + x._2.getTimes()(0)._2 / 1e9 + "\t" + x._2.getTimes()(0)._3 / 1e9)
      }
    })
    println(s"convolution\t$convTime")
    println(s"relu\t$reluTime")
  }

  "frcnn share convolution" should "work" in {
    val model = VggFRcnn(21, PostProcessParam(0.3f, 21, false, -1, 0))
    val sharedModel = SpatialShareConvolution.shareConvolution[Float](model)
    val target = Tensor(Storage(Array(0.0, 11.0, 0.0, 0.337411, 0.468211, 0.429096, 0.516061)
      .map(_.toFloat))).resize(1, 7)
    val frcnn = VggFRcnn(21, PostProcessParam(0.3f, 21, false, 100, 0.05))
    val criterion = FrcnnCriterion()
    val input = T()
    input.insert(Tensor[Float](1, 3, 600, 800).randn())
    input.insert(Tensor[Float](T(600, 800, 1, 1)).resize(1, 4))
    input.insert(target)

    model.loadModelWeights(sharedModel)

    model.forward(input)
    sharedModel.forward(input)

    val gradOut = Tensor[Float]().resizeAs(model.output.toTensor).randn()

    model.backward(input, gradOut)

    sharedModel.backward(input, gradOut)

    model.output should be(sharedModel.output)
    model.gradInput should be(sharedModel.gradInput)
  }

  "frcnn backward" should "work properly" in {
    TestUtil.middleRoot = "/home/jxy/data/middle/vgg16/new"
    val target = Tensor(Storage(Array(
      0, 15, 0,
      2.702702636718750000e+02,
      1.573573608398437500e+02,
      3.495495605468750000e+02,
      2.654654541015625000e+02,
      0, 16, 0,
      2.726726684570312500e+02,
      1.273273239135742188e+02,
      3.375375366210937500e+02,
      2.234234161376953125e+02,
      0, 16, 1,
      5.285285186767578125e+01,
      3.339339294433593750e+02,
      7.687687683105468750e+01,
      3.915915832519531250e+02)
      .map(_.toFloat))).resize(3, 7)
    VggFRcnn.debug = true
    val frcnn = Module.loadCaffe(VggFRcnn(21,
      PostProcessParam(0.3f, 21, false, -1, 0)),
      "/home/jxy/data/caffeModels/vgg16/test.prototxt",
      "/home/jxy/data/middle/vgg16/new/pretrained.caffemodel", false)
    val criterion = FrcnnCriterion()
    val (weights, grads) = frcnn.getParameters()
    val state = T(
      "learningRate" -> 0.001,
      "weightDecay" -> 0.0005,
      "momentum" -> 0.9,
      "dampening" -> 0.0,
      "learningRateSchedule" -> SGD.MultiStep(Array(80000, 100000, 120000), 0.1)
    )
    val sgd = new SGD[Float]

    val input = T()
    input.insert(TestUtil.loadFeatures("data"))
    input.insert(Tensor[Float](T(400, 601, 1.2012012, 1.2012012)).resize(1, 4))
    input.insert(target.clone())

    frcnn.zeroGradParameters()
    frcnn.forward(input)
    criterion.forward(frcnn.output.toTable, target)
    criterion.backward(frcnn.output.toTable, target)
    val gradInput = criterion.gradInput
    TestUtil.assertEqual("conv5_3", frcnn("relu5_3").get.output.toTensor[Float], 1e-4)
    TestUtil.assertEqual("rpn_cls_score", frcnn("rpn_cls_score").get.output.toTensor[Float], 1e-5)
    TestUtil.assertEqual("rpn_bbox_pred", frcnn("rpn_bbox_pred").get.output.toTensor[Float], 1e-5)
    TestUtil.assertEqual("cls_score", frcnn("cls_score").get.output.toTensor[Float], 1e-3)
    TestUtil.assertEqual("bbox_pred", frcnn("bbox_pred").get.output.toTensor[Float], 1e-3)
    TestUtil.assertEqual("rpn_rois", frcnn("proposal").get.output.toTensor[Float], 1e-1)

    TestUtil.assertEqual2(TestUtil.loadFeatures("rois"),
      frcnn("roi-data").get.output.toTable[Tensor[Float]](1), "rois", 1e-3)
    TestUtil.assertEqual2(TestUtil.loadFeatures("bbox_targets"),
      frcnn("roi-data").get.output.toTable[Tensor[Float]](3), "bbox_targets", 1e-3)

    TestUtil.assertEqual2(TestUtil.loadFeatures("rpn_labels").apply1(x =>
      if (x != -1) x + 1 else x),
      frcnn("rpn-data").get.output.toTable[Tensor[Float]](1), "rpn_labels", 1e-3)
    TestUtil.assertEqual2(TestUtil.loadFeatures("rpn_bbox_targets"),
      frcnn("rpn-data").get.output.toTable[Tensor[Float]](2), "rpn_bbox_targets", 1e-3)

    TestUtil.assertEqual("pool5", frcnn("pool5").get.output.toTensor[Float], 1e-5)

    TestUtil.assertEqual2(gradInput(3), TestUtil.loadFeatures("bbox_preddiff"), "bboxpred_diff",
      1e-6)
    TestUtil.assertEqual2(gradInput(4),
      TestUtil.loadFeatures("cls_scorediff"), "cls_scorediff", 1e-6)
    TestUtil.assertEqual2(gradInput(5),
      TestUtil.loadFeatures("rpn_cls_scorediff").resize(1, 2, 225, 38), "rpn_cls_scorediff", 1e-6)
    TestUtil.assertEqual2(gradInput(6),
      TestUtil.loadFeatures("rpn_bbox_preddiff"), "rpn_bbox_preddiff",
      1e-6)
    gradInput(4) = TestUtil.loadFeatures("cls_scorediff")
    gradInput(3) = TestUtil.loadFeatures("bbox_preddiff")
    gradInput(2) = null
    gradInput(1) = null
    gradInput(5) = TestUtil.loadFeatures("rpn_cls_scorediff")
    gradInput(6) = TestUtil.loadFeatures("rpn_bbox_preddiff")
    gradInput(7) = null
    frcnn.backward(input, criterion.gradInput)
    TestUtil.assertEqual2(TestUtil.loadFeatures("pool5diff").resize(128, 25088),
      frcnn("fc6").get.gradInput.toTensor[Float], "pool5diff", 1e-5)
    TestUtil.assertEqual2(TestUtil.loadFeatures("pool4diff"),
      frcnn("conv5_1").get.gradInput.toTensor[Float], "pool4diff", 1e-5)
//    TestUtil.assertEqual2(TestUtil.loadFeatures("fc7diff"),
//      frcnn("fc7").get.gradInput.toTensor[Float], "relu6", 1e-5)
    println(s"loss: ${criterion.output}")
  }

  "test table" should "work" in {
    val t = T()
    t.update(5, "2")
    t.update(9, "9")

    t(1) = "1"
    t(5) = "5"
    t(10) = "10"

    println(t)
  }

  "tile" should "work" in {
    val tensor = Tensor[Float](T(0.1, 0.1, 0.2, 0.2))
    val expand = tensor.reshape(Array(1, 4)).expand(Array(21, 4))
    println(expand)
    println(expand.isContiguous())
//    println(expand.reshape(Array(expand.nElement())))
    println(tensor)

  }
}

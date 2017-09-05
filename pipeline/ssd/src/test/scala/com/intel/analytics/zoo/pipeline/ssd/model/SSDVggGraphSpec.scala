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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.zoo.pipeline.common.nn.{MultiBoxLoss, MultiBoxLossParam}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

class SSDVggGraphSpec extends FlatSpec with Matchers {
  "SSD vgg 300 graph forward" should "work same as container model" in {
    val vgg300 = SSDVggSeq(300)
    val vgg300Graph = SSDVgg(300)
    vgg300Graph.loadModelWeights(vgg300)

    val input = Tensor[Float](1, 3, 300, 300).rand()

    vgg300.evaluate()
    vgg300Graph.evaluate()
    // warm up
    vgg300.forward(input)
    vgg300Graph.forward(input)

    var start = System.nanoTime()
    vgg300.forward(input)
    println("no graph takes " + (System.nanoTime() - start) / 1e9 + "s")

    start = System.nanoTime()
    vgg300Graph.forward(input)
    println(" graph takes " + (System.nanoTime() - start) / 1e9 + "s")


    vgg300.output.toTensor[Float].map(vgg300Graph.output.toTensor[Float], (a, b) => {
      assert(Math.abs(a - b) < 1e-7);
      a
    })
  }

  "share training" should "work properly" in {
    val target = Tensor(Storage(Array(0.0, 11.0, 0.0, 0.337411, 0.468211, 0.429096, 0.516061)
      .map(_.toFloat))).resize(1, 7)

    val vgg300 = SSDVggSeq(300)
    val vgg300Graph = SSDVgg(300)
    vgg300Graph.loadModelWeights(vgg300)

    val input = Tensor(1, 3, 300, 300).rand()
    val (weights, grads) = vgg300.getParameters()
    val (weightsGraph, gradGraph) = vgg300Graph.getParameters()
    val state = T(
      "learningRate" -> 0.001,
      "weightDecay" -> 0.0005,
      "momentum" -> 0.9,
      "dampening" -> 0.0,
      "learningRateSchedule" -> SGD.MultiStep(Array(80000, 100000, 120000), 0.1)
    )

    val state2 = T(
      "learningRate" -> 0.001,
      "weightDecay" -> 0.0005,
      "momentum" -> 0.9,
      "dampening" -> 0.0,
      "learningRateSchedule" -> SGD.MultiStep(Array(80000, 100000, 120000), 0.1)
    )
    vgg300.training()
    vgg300Graph.training()
    val criterion = new MultiBoxLoss(MultiBoxLossParam())
    val criterion2 = new MultiBoxLoss(MultiBoxLossParam())

    val sgd = new SGD[Float]

    vgg300.zeroGradParameters()
    vgg300.forward(input)
    criterion.forward(vgg300.output.toTable, target)
    println(s"loss: ${ criterion.output }")
    val gradOutputTest1 = criterion.backward(vgg300.output, target)
    vgg300.backward(input, gradOutputTest1)
    vgg300.zeroGradParameters()
    var start = System.nanoTime()
    val gradOutputTest4 = criterion.backward(vgg300.output, target)
    vgg300.backward(input, gradOutputTest4)
    println("no graph takes " + (System.nanoTime() - start) / 1e9 + "s")

    vgg300Graph.forward(input)
    criterion2.forward(vgg300Graph.output, target)
    println(s"loss2: ${ criterion2.output }")
    vgg300Graph.zeroGradParameters()

    val gradOutputTest2 = criterion2.backward(vgg300Graph.output, target)
    vgg300Graph.backward(input, gradOutputTest2)
    println("backward done")
    vgg300Graph.zeroGradParameters()

    start = System.nanoTime()
    val gradOutputTest3 = criterion2.backward(vgg300Graph.output, target)
    vgg300Graph.backward(input, gradOutputTest3)
    println("graph takes " + (System.nanoTime() - start) / 1e9 + "s")

    gradOutputTest2 should be(gradOutputTest3)

    vgg300.output.toTable(1).asInstanceOf[Tensor[Float]] should be
    (vgg300Graph.output.toTable(1).asInstanceOf[Tensor[Float]])

    vgg300.output.toTable(2).asInstanceOf[Tensor[Float]] should be
    (vgg300Graph.output.toTable(2).asInstanceOf[Tensor[Float]])

    vgg300.output.toTable(3).asInstanceOf[Tensor[Float]] should be
    (vgg300Graph.output.toTable(3).asInstanceOf[Tensor[Float]])
    criterion.output should be(criterion2.output)

    criterion.gradInput should be(criterion2.gradInput)

    vgg300Graph.gradInput.toTensor.map(vgg300.gradInput.toTensor, (a, b) => {
      assert(Math.abs(a - b) < 1e-6);
      a
    })

    val namedModule1 = vgg300.getParametersTable()
    val namedModule2 = vgg300Graph.getParametersTable()
    namedModule1.foreach(x => {
      namedModule2(x._1).asInstanceOf[Table] should be(x._2)
    })
  }
}

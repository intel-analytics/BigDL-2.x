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

package com.intel.analytics.bigdl.torch

import java.io.PrintWriter

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable.ArrayBuffer
import scala.math._
import scala.sys.process._

@com.intel.analytics.bigdl.tags.Serial
class LSTMPeepholeSpec  extends TorchSpec {
  override def torchCheck(): Unit = {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
    val tmpFile = java.io.File.createTempFile("checkRNN", ".lua")
    val writer = new PrintWriter(tmpFile)
    writer.write("exist = (pcall(require, 'rnn'))\n print(exist)")
    writer.close()

    val existsRNN =
      Seq(System.getProperty("torch_location", "th"), tmpFile.getAbsolutePath).!!.trim
    if (!existsRNN.contains("true")) {
      cancel("Torch rnn is not installed")
    }
  }

  "A LSTMPeephole" should " be fast" in {
    val inputSize = 300
    val hiddenSize = 300
    val batchSize = 12
    val time = 200
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](batchSize, time, inputSize).rand
    val gradOutput = Tensor[Float](batchSize, time, hiddenSize).rand

    val model = Recurrent[Float]()
        .add(LSTMPeephole[Float](inputSize, hiddenSize))

    var startTime = System.nanoTime()
    var duration = (System.nanoTime() - startTime) / 1e9
    var sum = 0.0

    println("warmup ..")
    for (i <- 1 to 5) {
      model.forward(input)
      model.backward(input, gradOutput)
    }

    val n = 5
    for (i <- 1 to n) {
      startTime = System.nanoTime()
      model.forward(input)
      model.backward(input, gradOutput)
      duration = (System.nanoTime() - startTime) / 1e9
      sum += duration
      println(s"iteration-${i}, elapsed time = ${duration}")
    }
    println(s"average elapsed time = ${sum / n}")
  }

  "A LSTMPeepwhole " should "has same loss as torch rnn" in {
    torchCheck()

    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 4
    val inputSize = 6
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 1

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(batchSize, seqLength, inputSize))
    val labels = Tensor[Double](Array(batchSize, seqLength))
    for (b <- 1 to batchSize) {
      for (i <- 1 to seqLength) {
        val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
        input.setValue(b, i, rdmInput, 1.0)
        val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0) * outputSize).toInt
        labels.setValue(b, i, rdmLabel)
      }
    }

    println(input)
    val rec = Recurrent[Double]()

    val model = Sequential[Double]()
      .add(rec
        .add(LSTMPeephole[Double](inputSize, hiddenSize)))
      //      .add(LSTM[Double](inputSize, hiddenSize)))
      //      .add(RnnCell[Double](inputSize, hiddenSize, Sigmoid[Double]())))
      //            .add(FastLSTMCell[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double](), false)
    val logSoftMax = TimeDistributed[Double](LogSoftMax[Double]())

    val (weights, grad) = model.getParameters()

    /*
     * Since we changed the structure of LSTMPeephole, we have to rearrange the parameters.
     */
    val (weightsArray, gradArray) = model.parameters()
    val weightsArrayTorch = weightsArray.clone

    val weightsTorch = new ArrayBuffer[Tensor[Double]]()

    val i2g2 = weightsArrayTorch(0).narrow(1, 1 + hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g2).copy(i2g2)
    val i2g2bias = weightsArrayTorch(1).narrow(1, 1 + hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g2bias).copy(i2g2bias)
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(2)).copy(weightsArrayTorch(2))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(3)).copy(weightsArrayTorch(3))
    val i2g1 = weightsArrayTorch(0).narrow(1, 1, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g1).copy(i2g1)
    val i2g1bias = weightsArrayTorch(1).narrow(1, 1, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g1bias).copy(i2g1bias)
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(4)).copy(weightsArrayTorch(4))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(5)).copy(weightsArrayTorch(5))

    val i2g3 = weightsArrayTorch(0).narrow(1, 1 + 2 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g3).copy(i2g3)
    val i2g3bias = weightsArrayTorch(1).narrow(1, 1 + 2 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g3bias).copy(i2g3bias)

    weightsTorch += Tensor().resizeAs(weightsArrayTorch(6)).copy(weightsArrayTorch(6))

    val i2g4 = weightsArrayTorch(0).narrow(1, 1 + 3 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g4).copy(i2g4)
    val i2g4bias = weightsArrayTorch(1).narrow(1, 1 + 3 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g4bias).copy(i2g4bias)

    weightsTorch += Tensor().resizeAs(weightsArrayTorch(7)).copy(weightsArrayTorch(7))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(8)).copy(weightsArrayTorch(8))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(9)).copy(weightsArrayTorch(9))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(10)).copy(weightsArrayTorch(10))

    val (weights2Torch, grad2Torch) =
      (Module.flatten[Double](weightsTorch.toArray), Module.flatten[Double](gradArray))

    val code =
      s"""
         |
      |-- 1.4. Combine 1.1 and 1.3 to produce final model
         |require 'rnn'
         |torch.manualSeed($seed)
         |
      |local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
         |         :add(nn.ParallelTable()
         |            :add(nn.Linear($inputSize, $hiddenSize)) -- input layer
         |            :add(nn.Linear($hiddenSize, $hiddenSize))) -- recurrent layer
         |         :add(nn.CAddTable()) -- merge
         |         :add(nn.Sigmoid()) -- transfer
         |     --    :add(nn.Tanh()) -- transfer
         |
      | -- local rm1 =  nn.Sequential() -- input is {x[t], h[t-1]}
         | --        :add(nn.ParallelTable()
         | --           :add(nn.Linear($inputSize, $hiddenSize)) -- input layer
         | --           :add(nn.Identity())) -- recurrent layer
         | --        :add(nn.CAddTable()) -- merge
         | --        :add(nn.Sigmoid()) -- transfer
         |      rnn = nn.Recurrence(rm, $hiddenSize, 1)
         | --     rnn.userPrevOutput = torch.Tensor(1, $hiddenSize):zero()
         |
      |model = nn.Sequential()
         |:add(nn.SplitTable(1))
         |:add(nn.Sequencer(
         | nn.Sequential()
         |   :add(nn.LSTM($inputSize, $hiddenSize, 1, true))
         |   --:add(nn.FastLSTM($inputSize, $hiddenSize, 1, nil, nil, nil, 0.3))
         |   :add(nn.Linear($hiddenSize, $outputSize))
         |   ))
         |
         |
         |local parameters, gradParameters = model:getParameters()
         |model:zeroGradParameters()
         |parameters:copy(weights)
         |
         |parameters_initial = parameters : clone()
         |gradParameters_initial = gradParameters : clone()
         |
         |local criterion =  nn.SequencerCriterion(nn.CrossEntropyCriterion())
         |
      |
      |state = {
         |  learningRate = 0.5,
         |  momentum = 0.0,
         |  dampening = 0.0,
         |  weightDecay = 0.0
         |}
         |
      |feval = function(x)
         |model:zeroGradParameters()
         |model_initial = model : clone()
         |
      |local output1 = model:forward(input)
         |local err1 = criterion:forward(output1, labels)
         |local gradOutput1 = criterion:backward(output1, labels)
         |model:backward(input, gradOutput1)
         |return err1, gradParameters
         |end
         |
      |for i = 1,100,1 do
         |   optim.sgd(feval, parameters, state)
         |end
         |
      |output=model.output
         |err=criterion.output
         |err2=criterion.gradInput
         |gradOutput=criterion.gradInput
         |gradInput = model.gradInput
    """.stripMargin

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input.transpose(1, 2), "weights" -> weights2Torch,
        "labels" -> SplitTable[Double](1).forward(labels.t())),
      Array("output", "err", "parameters", "gradParameters", "output2", "gradInput", "err2"))

    //    println("Element forward: " + output1)
    //    println("BigDL forward: " + model.forward(input).toTensor[Double].clone())
    //
    val luaOutput2 = torchResult("err").asInstanceOf[Double]
    val luaweight = torchResult("parameters").asInstanceOf[Tensor[Double]]

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    val start = System.nanoTime()
    var loss: Array[Double] = null
    for (i <- 1 to 100) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    //    println("Element weight: " + luaweight)
    //    println("BigDL weight: " + weights)
    //
    //    println("Element forward: " + output1)
    //    println("BigDL forward: " + model.output)
    //
    //    println("BigDL labels: " + labels)
    //
    //    val crtnGradInput = criterion.backward(model.output, labels)
    //    println(s"element: criterion gradInput: $err2")
    //    println("BigDL criterion gradInput: " + crtnGradInput)
    //
    //    println(s"element: gradInput: $gradInput1")
    //    println("BigDL: " + model.gradInput.toTensor[Double])
    //
    //    println(s"element: gradWeight: $gradParameters")
    //    println("BigDL: " + model.getParameters()._2)


    val output = model.forward(input).toTensor
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(3)._2

    luaOutput2 should be(loss(0) +- 1e-5)
  }


  "A LSTMPeepwhole " should "converge" in {
    torchCheck()

    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 4
    val inputSize = 6
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 100

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(1, seqLength, inputSize))
    val labels = Tensor[Double](Array(1, seqLength))
    for (i <- 1 to seqLength) {
      val rdmLabel = Math.ceil(RNG.uniform(0, 1) * outputSize).toInt
      val rdmInput = Math.ceil(RNG.uniform(0, 1) * inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    println(input)
    val rec = Recurrent[Double]()

    val model = Sequential[Double]()
      .add(rec
        .add(LSTMPeephole[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double](), false)
    val logSoftMax = TimeDistributed[Double](LogSoftMax[Double]())

    val (weights, grad) = model.getParameters()

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    val start = System.nanoTime()
    var loss: Array[Double] = null
    for (i <- 1 to 100) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    val output = model.forward(input).toTensor
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(3)._2

    labels.squeeze() should be (prediction.squeeze())
  }


  "A LSTMPeepwhole " should "has same loss as torch rnn in batch mode" in {
    torchCheck()

    import com.intel.analytics.bigdl.numeric.NumericFloat
    val hiddenSize = 4
    val inputSize = 6
    val outputSize = 5
    val bpttTruncate = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 5

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(batchSize, seqLength, inputSize))
    val labels = Tensor[Double](Array(batchSize, seqLength))
    for (b <- 1 to batchSize) {
      for (i <- 1 to seqLength) {
        val rdmInput = Math.ceil(RNG.uniform(0, 1)  * inputSize).toInt
        input.setValue(b, i, rdmInput, 1f)
        val rdmLabel = Math.ceil(RNG.uniform(0, 1)  * outputSize).toInt
        labels.setValue(b, i, rdmLabel)
      }
    }

    println(input)
    val rec = Recurrent()

    val model = Sequential()
      .add(rec
        .add(LSTMPeephole(inputSize, hiddenSize)))
      .add(TimeDistributed(Linear(hiddenSize, outputSize)))

    val criterion = TimeDistributedCriterion(
      CrossEntropyCriterion())
    val logSoftMax = TimeDistributed(LogSoftMax())

    val (weights, grad) = model.getParameters()


    /*
     * Since we changed the structure of LSTMPeephole, we have to rearrange the parameters.
     */
    val (weightsArray, gradArray) = model.parameters()
    val weightsArrayTorch = weightsArray.clone

    val weightsTorch = new ArrayBuffer[Tensor[Float]]()

    val i2g2 = weightsArrayTorch(0).narrow(1, 1 + hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g2).copy(i2g2)
    val i2g2bias = weightsArrayTorch(1).narrow(1, 1 + hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g2bias).copy(i2g2bias)
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(2)).copy(weightsArrayTorch(2))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(3)).copy(weightsArrayTorch(3))
    val i2g1 = weightsArrayTorch(0).narrow(1, 1, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g1).copy(i2g1)
    val i2g1bias = weightsArrayTorch(1).narrow(1, 1, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g1bias).copy(i2g1bias)
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(4)).copy(weightsArrayTorch(4))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(5)).copy(weightsArrayTorch(5))

    val i2g3 = weightsArrayTorch(0).narrow(1, 1 + 2 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g3).copy(i2g3)
    val i2g3bias = weightsArrayTorch(1).narrow(1, 1 + 2 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g3bias).copy(i2g3bias)

    weightsTorch += Tensor().resizeAs(weightsArrayTorch(6)).copy(weightsArrayTorch(6))

    val i2g4 = weightsArrayTorch(0).narrow(1, 1 + 3 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g4).copy(i2g4)
    val i2g4bias = weightsArrayTorch(1).narrow(1, 1 + 3 * hiddenSize, hiddenSize)
    weightsTorch += Tensor().resizeAs(i2g4bias).copy(i2g4bias)

    weightsTorch += Tensor().resizeAs(weightsArrayTorch(7)).copy(weightsArrayTorch(7))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(8)).copy(weightsArrayTorch(8))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(9)).copy(weightsArrayTorch(9))
    weightsTorch += Tensor().resizeAs(weightsArrayTorch(10)).copy(weightsArrayTorch(10))

    val (weights2Torch, grad2Torch) =
      (Module.flatten[Float](weightsTorch.toArray), Module.flatten[Float](gradArray))

    val code =
      s"""
         |
      |-- 1.4. Combine 1.1 and 1.3 to produce final model
         |require 'rnn'
         |torch.manualSeed($seed)
         |
      |local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
         |         :add(nn.ParallelTable()
         |            :add(nn.Linear($inputSize, $hiddenSize)) -- input layer
         |            :add(nn.Linear($hiddenSize, $hiddenSize))) -- recurrent layer
         |         :add(nn.CAddTable()) -- merge
         |         :add(nn.Sigmoid()) -- transfer
         |     --    :add(nn.Tanh()) -- transfer
         |
      |model = nn.Sequential()
         |:add(nn.SplitTable(1))
         |:add(nn.Sequencer(
         | nn.Sequential()
         |   :add(nn.LSTM($inputSize, $hiddenSize, 1, true))
         |   --:add(nn.FastLSTM($inputSize, $hiddenSize, 1))
         |   :add(nn.Linear($hiddenSize, $outputSize))
         |   ))
         |
         |
         |local parameters, gradParameters = model:getParameters()
         |model:zeroGradParameters()
         |parameters:copy(weights)
         |
         |parameters_initial = parameters : clone()
         |gradParameters_initial = gradParameters : clone()
         |
         |local criterion =  nn.SequencerCriterion(nn.CrossEntropyCriterion())
         |
      |
      |state = {
         |  learningRate = 0.5,
         |  momentum = 0.0,
         |  dampening = 0.0,
         |  weightDecay = 0.0
         |}
         |
      |feval = function(x)
         |model:zeroGradParameters()
         |model_initial = model : clone()
         |
      |local output1 = model:forward(input)
         |local err1 = criterion:forward(output1, labels)
         |local gradOutput1 = criterion:backward(output1, labels)
         |model:backward(input, gradOutput1)
         |return err1, gradParameters
         |end
         |
      |for i = 1,20,1 do
         |   optim.sgd(feval, parameters, state)
         |end
         |
      |output=model.output
         |err=criterion.output
         |err2=criterion.gradInput
         |gradOutput=criterion.gradInput
         |gradInput = model.gradInput
    """.stripMargin
    scala.Seq

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input.transpose(1, 2), "weights" -> weights2Torch,
        "labels" -> SplitTable[Double](1).forward(labels.t())),
      Array("output", "err", "parameters", "gradParameters", "output2", "gradInput", "err2"))

    //    println("Element forward: " + output1)
    //    println("BigDL forward: " + model.forward(input).toTensor[Double].clone())
    //
    val luaOutput2 = torchResult("err").asInstanceOf[Double]
    val luaweight = torchResult("parameters").asInstanceOf[Tensor[Double]]

    val floatInput = Tensor(Array(batchSize, seqLength, inputSize))
    val floatLabel = Tensor(Array(batchSize, seqLength))

    for (i <- 0 until floatInput.nElement()) {
      floatInput.storage().array()(i) = input.storage().array()(i).toFloat
    }
    for (i <- 0 until floatLabel.nElement()) {
      floatLabel.storage().array()(i) = labels.storage().array()(i).toFloat
    }

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Float]()
    def feval(x: Tensor[Float]): (Float, Tensor[Float]) = {
      val output = model.forward(floatInput).asInstanceOf[Tensor[Float]]
      val _loss = criterion.forward(output, floatLabel)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, floatLabel)
      model.backward(floatInput, gradInput)
      (_loss, grad)
    }

    val start = System.nanoTime()
    var loss: Array[Float] = null
    for (i <- 1 to 20) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    //    println("Element weight: " + luaweight)
    //    println("BigDL weight: " + weights)
    //
    //    println("Element forward: " + output1)
    //    println("BigDL forward: " + model.output)
    //
    //    println("BigDL labels: " + labels)
    //
    //    val crtnGradInput = criterion.backward(model.output, labels)
    //    println(s"element: criterion gradInput: $err2")
    //    println("BigDL criterion gradInput: " + crtnGradInput)
    //
    //    println(s"element: gradInput: $gradInput1")
    //    println("BigDL: " + model.gradInput.toTensor[Double])
    //
    //    println(s"element: gradWeight: $gradParameters")
    //    println("BigDL: " + model.getParameters()._2)


    val output = model.forward(floatInput).toTensor
    val logOutput = logSoftMax.forward(output)

    luaOutput2.toFloat should be(loss(0) +- 2e-2f)
  }


  "A LSTMPeepwhole " should "converge in batch mode" in {
    torchCheck()

    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 4
    val inputSize = 6
    val outputSize = 5
    val bpttTruncate = 3
    val batchSize = 3
    val seqLength = 5
    val seed = 100

    RNG.setSeed(seed)
    val input = Tensor[Double](Array(batchSize, seqLength, inputSize))
    val labels = Tensor[Double](Array(batchSize, seqLength))
    for (b <- 1 to batchSize) {
      for (i <- 1 to seqLength) {
        val rdmInput = Math.ceil(RNG.uniform(0.0, 1.0) * inputSize).toInt
        input.setValue(b, i, rdmInput, 1.0)
        val rdmLabel = Math.ceil(RNG.uniform(0.0, 1.0) * outputSize).toInt
        labels.setValue(b, i, rdmLabel)
      }
    }

    println(input)
    val rec = Recurrent[Double]()

    val model = Sequential[Double]()
      .add(rec
        .add(LSTMPeephole[Double](inputSize, hiddenSize)))
      .add(TimeDistributed[Double](Linear[Double](hiddenSize, outputSize)))

    val criterion = TimeDistributedCriterion[Double](
      CrossEntropyCriterion[Double](), false)
    val logSoftMax = TimeDistributed[Double](LogSoftMax[Double]())

    val (weights, grad) = model.getParameters()

    val state = T("learningRate" -> 0.5, "momentum" -> 0.0,
      "weightDecay" -> 0.0, "dampening" -> 0.0)
    val sgd = new SGD[Double]
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      val output = model.forward(input).asInstanceOf[Tensor[Double]]
      val _loss = criterion.forward(output, labels)
      model.zeroGradParameters()
      val gradInput = criterion.backward(output, labels)
      model.backward(input, gradInput)
      (_loss, grad)
    }

    val start = System.nanoTime()
    var loss: Array[Double] = null
    for (i <- 1 to 300) {
      loss = sgd.optimize(feval, weights, state)._2
      println(s"${i}-th loss = ${loss(0)}")
    }
    val end = System.nanoTime()
    println("Time: " + (end - start) / 1E6)

    val output = model.forward(input).toTensor
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(3)._2

    labels.squeeze() should be (prediction.squeeze())
  }

  "A LSTMPeepwhole " should "perform correct gradient check" in {
    torchCheck()

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    RNG.setSeed(seed)

    val model = Sequential[Double]()
      .add(Recurrent[Double]()
        .add(LSTMPeephole[Double](inputSize, hiddenSize)))
      .add(Select(1, 1))
      .add(Linear[Double](hiddenSize, outputSize))

    model.reset()

    val input = Tensor[Double](Array(1, 5, inputSize))
    val labels = Tensor[Double](Array(1, 5))
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(RNG.uniform(0, 1)*inputSize).toInt
      val rdmInput = Math.ceil(RNG.uniform(0, 1)*inputSize).toInt
      input.setValue(1, i, rdmInput, 1.0)
      labels.setValue(1, i, rdmLabel)
    }

    println("gradient check for input")
    val gradCheckerInput = new GradientChecker(1e-2, 1)
    val checkFlagInput = gradCheckerInput.checkLayer[Double](model, input)
    println("gradient check for weights")
    val gradCheck = new GradientCheckerRNN(1e-2, 1)
    val checkFlag = gradCheck.checkLayer(model, input, labels)
  }

  "A LSTMPeephole" should " be able to return finalstate" in {
    val inputSize = 10
    val hiddenSize = 30
    val batchSize = 12
    val time = 20
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](batchSize, time, inputSize).rand
    val gradOutput = Tensor[Float](batchSize, time, hiddenSize).rand

    val model = Recurrent[Float]()
      .add(LSTMPeephole[Float](inputSize, hiddenSize))

    var output: Activity = null
    for (i <- 1 to 5) {
      output = model.forward(input)
      model.backward(input, gradOutput)
    }
    val (finalState, cellStatus) = model.getFinalStateAndCellStatus()
    finalState.map(output.asInstanceOf[Tensor[Float]].select(2, time), (v1, v2) => {
      assert(abs(v1 - v2) == 0)
      v1
    })
  }
}

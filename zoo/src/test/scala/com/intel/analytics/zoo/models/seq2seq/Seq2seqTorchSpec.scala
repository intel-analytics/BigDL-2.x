/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.zoo.models.common.TH
import com.intel.analytics.zoo.models.seq2seq.Seq2seq
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._

class Seq2seqTorchSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "A Seq2seq" should "generate the same result with torch " in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 2
    val inputSize = 5
    val seed = 123

    RNG.setSeed(seed)
    val encInSeq = Array(1.0, 2, 3, 3, 2, 1)
    val decInSeq = Array(1.0, 2, 3, 4, 4, 3, 2, 1)
    val decOutData = Array(2.0, 3, 4, 1, 1, 2, 4, 3)
    val input1 = Tensor[Double](encInSeq, Array(2, 3))
    val input2 = Tensor[Double](decInSeq, Array(2, 4))
    val input = T(input1, input2)
    val decOutSeq = Tensor[Double](decOutData, Array(2, 4))

    val enclstm = LSTM[Double](
      hiddenSize,
      hiddenSize).asInstanceOf[Cell[Double]]
    val encoderCell = Array(enclstm)
    val (enclstmW, enclstmG) = Recurrent[Double]().add(enclstm).getParameters()

    val declstm = LSTM[Double](
      hiddenSize,
      hiddenSize).asInstanceOf[Cell[Double]]
    val decoderCell = Array(declstm)
    val (declstmW, declstmG) = Recurrent[Double]().add(declstm).getParameters()

    val enclookuptable = LookupTable(inputSize, hiddenSize)
    val (enclookuptableW, enclookuptableG) = enclookuptable.getParameters()

    val declookuptable = LookupTable(inputSize, hiddenSize)
    val (declookuptableW, declookuptableG) = declookuptable.getParameters()

    val linear = Linear(hiddenSize, inputSize)
    val (linearW, linearG) = linear.getParameters()

    val code =
      s"""
         |
      |-- 1.4. Combine 1.1 and 1.3 to produce final model
         |require 'rnn'
         |
         |local function forwardConnect(encLSTM, decLSTM)
         |    decLSTM.userPrevOutput =
         |      nn.rnn.recursiveCopy(decLSTM.userPrevOutput, encLSTM.outputs[3])
         |    decLSTM.userPrevCell = nn.rnn.recursiveCopy(decLSTM.userPrevCell, encLSTM.cells[3])
         |end
         |
         |local function backwardConnect(encLSTM, decLSTM)
         |    encLSTM:setGradHiddenState(3, decLSTM:getGradHiddenState(0))
         |end
         |local enc1 = nn.Sequential()
         |lookup = nn.LookupTable(5, 2)
         |local parameters, gradParameters = lookup:getParameters()
         |parameters:copy(lookuptableP1)
         |enc1:add(lookup)
         |enc1:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
         |enc1output = enc1:forward(input)
         |enc1:zeroGradParameters()
         |
         |local enc = nn.Sequential()
         |local encLSTM = nn.FastLSTM(2, 2)
         |enc:add(nn.Sequencer(encLSTM))
         |enc:add(nn.SelectTable(-1))
         |local parameters2, gradParameters2 = enc:getParameters()
         |parameters2:copy(enclstm)
         |enc:zeroGradParameters()
         |
         |local dec = nn.Sequential()
         |lookup2 = nn.LookupTable(5, 2)
         |local parameters3, gradParameters3 = lookup2:getParameters()
         |parameters3:copy(lookuptableP2)
         |dec:add(lookup2)
         |dec:add(nn.SplitTable(1, 2)) -- works for both online and mini-batch mode
         |local decLSTM = nn.FastLSTM(2, 2)
         |local parameters4, gradParameters4 = decLSTM:getParameters()
         |parameters4:copy(declstm)
         |dec:add(nn.Sequencer(decLSTM))
         |linear2 = nn.Linear(2, 5)
         |dec:add(nn.Sequencer(linear2))
         |local parameters5, gradParameters5 = linear2:getParameters()
         |parameters5:copy(linear2P)
         |dec:add(nn.Sequencer(nn.LogSoftMax()))
         |dec:zeroGradParameters()
         |
         |local encOutput = enc:forward(enc1output)
         |forwardConnect(encLSTM, decLSTM)
         |local decOutput = dec:forward(input2)
         |local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
         |local decOut = nn.SplitTable(1, 1):forward(decOutSeq)
         |local Edec = criterion:forward(decOutput, decOut)
         |
         |local gEdec = criterion:backward(decOutput, decOut)
         |local decGradInput = dec:backward(input2, gEdec)
         |backwardConnect(encLSTM, decLSTM)
         |local zeroTensor = torch.zeros(encOutput:size())
         |local encGradInput = enc:backward(enc1output, zeroTensor)
         |local enc1GradInput = enc1:backward(input, encGradInput)
    """.stripMargin
    scala.Seq

    val (luaTime, torchResult) = TH.run(code,
      Map("input" -> input1, "lookuptableP1" -> enclookuptableW, "enclstm" -> enclstmW,
        "input2" -> input2, "lookuptableP2" -> declookuptableW, "declstm" -> declstmW,
        "decOutSeq" -> decOutSeq, "linear2P" -> linearW),
      Array("gradParameters2", "decOutput", "Edec",
        "encGradInput", "gEdec", "gradParameters4"))

    val luaOutput = torchResult("decOutput").asInstanceOf[Table]
    val luaLoss = torchResult("Edec").asInstanceOf[Double]
    val luagEdc = torchResult("gEdec").asInstanceOf[Table]
    val luaencGradInput = torchResult("encGradInput").asInstanceOf[Table]
    val luaenclstmG = torchResult("gradParameters2").asInstanceOf[Tensor[Double]]
    val luadeclstmG = torchResult("gradParameters4").asInstanceOf[Tensor[Double]]

    val preEncoder = enclookuptable
    val preDecoder = declookuptable
    val seq2seq = Seq2seq(encoderCell, decoderCell, preEncoder = preEncoder,
      preDecoder = preDecoder)

    val model = Sequential().add(seq2seq).add(TimeDistributed(linear))
      .add(TimeDistributed(LogSoftMax()))
    val output = model.forward(input).toTensor
    for (i <- 1 to 4) {
      require(output.select(2, i).almostEqual(luaOutput.toTable[Tensor[Double]](i), 1e-8) == true)
    }

    val criterion = TimeDistributedCriterion(ClassNLLCriterion())
    val loss = criterion.forward(output, decOutSeq)
    assert(abs(loss - luaLoss) <= 1e-8)

    val gEdec = criterion.backward(output, decOutSeq)
    for (i <- 1 to 4) {
      gEdec.select(2, i).almostEqual(luagEdc.toTable[Tensor[Double]](i), 1e-8)
    }
    model.backward(input, gEdec)
    val encRec = seq2seq.encoder.modules.apply(1)
    for (i <- 1 to 3) {
      luaencGradInput.toTable[Tensor[Double]](i)
        .almostEqual(encRec.gradInput.toTensor.select(2, i), 1e-8)
    }
    require(luaenclstmG.almostEqual(enclstmG, 1e-8) == true)
    require(luadeclstmG.almostEqual(declstmG, 1e-8) == true)
  }
}

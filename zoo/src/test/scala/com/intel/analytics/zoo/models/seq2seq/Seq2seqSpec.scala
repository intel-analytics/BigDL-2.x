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

package com.intel.analytics.zoo.models.seq2seq


import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.numeric.NumericDouble
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Shape, T}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

class Seq2seqSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "A Seq2seq" should "work with PassThroughBridge" in {
    val nbKernal = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand

    val encoderCells = Array(ConvLSTM2D[Double](
      7,
      nbKernal,
      inputShape = Shape(input.size())), ConvLSTM2D[Double](
      12,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 7, 5, 5)), ConvLSTM2D[Double](
      3,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 12, 5, 5))).asInstanceOf[Array[Recurrent[Double]]]

    val decoderCells = Array(ConvLSTM2D[Double](
      7,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 3, 5, 5)), ConvLSTM2D[Double](
      12,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 7, 5, 5)), ConvLSTM2D[Double](
      3,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 12, 5, 5))).asInstanceOf[Array[Recurrent[Double]]]

    val model = Seq2seq(encoderCells, decoderCells)

    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    val output = model.inference(T(input, input.narrow(2, input.size(2), 1)),
      maxSeqLen = 9).toTensor
    assert(output.size(2) == 10)
  }

  "A Seq2seq" should "work with ZeroBridge" in {
    val nbKernal = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand

    val encoderCells = Array(ConvLSTM2D[Double](
      7,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 3, 5, 5)), ConvLSTM2D[Double](
      12,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 7, 5, 5)), ConvLSTM2D[Double](
      3,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 12, 5, 5))).asInstanceOf[Array[Recurrent[Double]]]

    val decoderCells = Array(ConvLSTM2D[Double](
      7,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 3, 5, 5)), ConvLSTM2D[Double](
      12,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 7, 5, 5)), ConvLSTM2D[Double](
      3,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 12, 5, 5))).asInstanceOf[Array[Recurrent[Double]]]

    val model = Seq2seq(encoderCells, decoderCells, bridges = new ZeroBridge())

    model.parameters()
    model.getParametersTable()
    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    val output = model.inference(T(input, input.narrow(2, input.size(2), 1)),
      maxSeqLen = 7).toTensor
    assert(output.size(2) == 8)
  }

  "A Seq2seq" should "work with InitialStateBridge" in {
    val nbKernal = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand

    val encoderCells = Array(ConvLSTM2D[Double](
      7,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 3, 5, 5)), ConvLSTM2D[Double](
      12,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 7, 5, 5)), ConvLSTM2D[Double](
      3,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 12, 5, 5))).asInstanceOf[Array[Recurrent[Double]]]

    val decoderCells = Array(ConvLSTM2D[Double](
      14,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 3, 5, 5)), ConvLSTM2D[Double](
      25,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 14, 5, 5)), ConvLSTM2D[Double](
      3,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 25, 5, 5))).asInstanceOf[Array[Recurrent[Double]]]

    val activations = Array(
      Array(new Convolution2D[Double](14, nbKernal, nbKernal, pads = Array(nbKernal/2, nbKernal/2),
        inputShape = Shape(batchSize, seqLength, 7, 5, 5)),
      new Convolution2D[Double](14, nbKernal, nbKernal, pads = Array(nbKernal/2, nbKernal/2),
        inputShape = Shape(batchSize, seqLength, 7, 5, 5))),
      Array(new Convolution2D[Double](25, nbKernal, nbKernal, pads = Array(nbKernal/2, nbKernal/2),
        inputShape = Shape(batchSize, seqLength, 12, 5, 5))),
      new Convolution2D[Double](25, nbKernal, nbKernal, pads = Array(nbKernal/2, nbKernal/2),
          inputShape = Shape(batchSize, seqLength, 12, 5, 5)), null
    ).asInstanceOf[Array[Array[KerasLayer[Tensor[Double], Tensor[Double], Double]]]]
    val model = Seq2seq(encoderCells, decoderCells,
      bridges = new InitialStateBridge[Double](activations))
    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    val output = model.inference(T(input, input.narrow(2, input.size(2), 1)),
      maxSeqLen = 10).toTensor
    assert(output.size(2) == 11)
  }

//  "A Seq2seq" should "work with InitialStateBridge2" in {
//    val kernalW = 3
//    val kernalH = 3
//    val seqLength = 5
//    val seed = 100
//    val batchSize = 4
//
//    RNG.setSeed(seed)
//    val input = Tensor[Double](batchSize, seqLength, 3, 5, 5, 5).rand
//    val decoderInput = Tensor[Double](batchSize, 1, 5, 5, 5, 5).rand
//    val gradOutput = Tensor[Double](batchSize, seqLength, 5, 5, 5, 5).rand
//
//    val encoderCells = Array(ConvLSTMPeephole3D[Double](
//      3,
//      7,
//      kernalW, kernalH,
//      1), ConvLSTMPeephole3D[Double](
//      7,
//      7,
//      kernalW, kernalH,
//      1), ConvLSTMPeephole3D[Double](
//      7,
//      7,
//      kernalW, kernalH,
//      1)).asInstanceOf[Array[Cell[Double]]]
//
//    val decoderCells = Array(ConvLSTMPeephole3D[Double](
//      5,
//      5,
//      kernalW, kernalH,
//      1), ConvLSTMPeephole3D[Double](
//      5,
//      5,
//      kernalW, kernalH,
//      1), ConvLSTMPeephole3D[Double](
//      5,
//      5,
//      kernalW, kernalH,
//      1)).asInstanceOf[Array[Cell[Double]]]
//
//    val activations = Array(
//      Array(VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1),
//        VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1)),
//      Array(VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1),
//        VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1)),
//      Array(VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1),
//        VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1))
//    ).asInstanceOf[Array[Array[TensorModule[Double]]]]
//
//    val preDecoder = Sequential().add(Contiguous())
//      .add(TimeDistributed(VolumetricConvolution[Double](3, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1)))
//    val model = Seq2seq(encoderCells, decoderCells, preDecoder = preDecoder,
//      bridges = new InitialStateBridge[Double](activations))
//
//    for (i <- 0 until 3) {
//      model.forward(T(input, input))
//      model.backward(T(input, input), gradOutput)
//    }
//
//    val output = model.inference(T(input, decoderInput),
//      maxSeqLen = 15)
//    assert(output.size(2) == 16)
//  }

  "A Seq2seq" should "work with single cell" in {
    val hiddenSize = 7
    val inputSize = 7
    val nbKernal = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, inputSize, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 5, 5).rand

    val encoderCells = Array(ConvLSTM2D[Double](
      hiddenSize,
      nbKernal,
      inputShape = Shape(input.size()))).asInstanceOf[Array[Recurrent[Double]]]

    val decoderCells = Array(ConvLSTM2D[Double](
      hiddenSize,
      nbKernal,
      inputShape = Shape(input.size()))).asInstanceOf[Array[Recurrent[Double]]]

    val model = Seq2seq(encoderCells, decoderCells)

    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    val output = model.inference(T(input, input.narrow(2, input.size(2), 1)),
      maxSeqLen = 12).toTensor
    assert(output.size(2) == 13)
  }

//  "A Seq2seq" should "work with getParameters" in {
//    val inputSize = 2
//    val hiddenSize = 16
//    val outputSize = 1
//    val seed = 100
//
//    RNG.setSeed(seed)
//
//    val encoderCells = Array(ConvLSTMPeephole3D[Double](
//      inputSize, hiddenSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]
//
//    val decoderCells = Array(ConvLSTMPeephole3D[Double](
//      outputSize, outputSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]
//
//    val activations = Array(
//      Array(VolumetricConvolution[Double](hiddenSize, outputSize, 3, 3, 3, 1, 1, 1, -1, -1, -1)),
//      Array(VolumetricConvolution[Double](hiddenSize, outputSize, 3, 3, 3, 1, 1, 1, -1, -1, -1))
//    ).asInstanceOf[Array[Array[TensorModule[Double]]]]
//
//    val _bridges = new InitialStateBridge[Double](activations)
//    val _preDecoder = Sequential().add(Contiguous())
//    .add(VolumetricConvolution(inputSize, outputSize, 3, 3, 3, 1, 1, 1, -1, -1, -1))
//
//    val model = Seq2seq(encoderCells, decoderCells, bridges = _bridges, preDecoder = _preDecoder)
//
//    require(model.getParametersTable().length() == 19)
//    require(model.parameters()._1.length == 30)
//  }
//
//  "A Seq2seq" should "work with getParameters 2" in {
//    val inputSize = 2
//    val hiddenSize = 16
//    val outputSize = 1
//    val seed = 100
//
//    RNG.setSeed(seed)
//
//    val encoderCells = Array(ConvLSTMPeephole3D[Double](
//      inputSize, hiddenSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]
//
//    val decoderCells = Array(ConvLSTMPeephole3D[Double](
//      outputSize, outputSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]
//
//    val model = Seq2seq(encoderCells, decoderCells)
//
//    val encoderCells2 = Array(ConvLSTMPeephole3D[Double](
//      inputSize, hiddenSize, 3, 3, withPeephole = false),
//      ConvLSTMPeephole3D[Double](
//        inputSize, hiddenSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]
//
//    val decoderCells2 = Array(ConvLSTMPeephole3D[Double](
//      inputSize, hiddenSize, 3, 3, withPeephole = false),
//      ConvLSTMPeephole3D[Double](
//        inputSize, hiddenSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]
//
//    val model2 = Seq2seq(encoderCells2, decoderCells2)
//    require(2 * model.getParametersTable().length() == model2.getParametersTable().length())
//  }

  "A Seq2seq" should "work with stop sign" in {
    val hiddenSize = 7
    val inputSize = 7
    val nbKernal = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, inputSize, 5, 5)

    val encoderCells = Array(ConvLSTM2D[Double](
      hiddenSize,
      nbKernal,
      inputShape = Shape(input.size()))).asInstanceOf[Array[Recurrent[Double]]]

    val decoderCells = Array(ConvLSTM2D[Double](
      hiddenSize,
      nbKernal,
      inputShape = Shape(input.size()))).asInstanceOf[Array[Recurrent[Double]]]

    val model = Seq2seq(encoderCells, decoderCells)

    var output = model.inference(T(input, input.narrow(2, seqLength, 1)),
      maxSeqLen = seqLength).toTensor
    require(output.size(2) == seqLength + 1)

    model.parameters()._1.foreach(_.fill(0.0))
    output = model.inference(T(input, input.narrow(2, seqLength, 1)),
      stopSign = Tensor[Double](batchSize, hiddenSize, 5, 5)).toTensor
    require(output.size(2) == 2)
  }

  "A Seq2seq" should "work with generator" in {
    val hiddenSize = 7
    val inputSize = 7
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, inputSize)
    val decoderInput = Tensor[Double](batchSize, 1, 5)
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize)

    val encoderCells = Array(LSTM[Double](hiddenSize,  returnSequences = true,
      inputShape = Shape(seqLength, inputSize))).asInstanceOf[Array[Recurrent[Double]]]

    val decoderCells = Array(LSTM[Double](hiddenSize,  returnSequences = true,
      inputShape = Shape(1, 5))).asInstanceOf[Array[Recurrent[Double]]]

    val model = Seq2seq(encoderCells, decoderCells)

    for (i <- 0 until 3) {
      model.forward(T(input, decoderInput)).toTensor
      model.backward(T(input, decoderInput), gradOutput)
    }

    val layer = Sequential[Double]().add(
      TimeDistributed[Double](Dense[Double](5), inputShape = Shape(batchSize, hiddenSize)))
    val output = model.inference(T(input, decoderInput),
      maxSeqLen = seqLength,
      infer = layer.asInstanceOf[KerasLayer[Tensor[Double], Tensor[Double], Double]]).toTensor
    require(output.size(2) == seqLength + 1)
  }

  "A Seq2seq serialize" should "work" in {
    val nbKernal = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand

    val encoderCells = Array(ConvLSTM2D[Double](
      7,
      nbKernal,
      inputShape = Shape(input.size())), ConvLSTM2D[Double](
      12,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 7, 5, 5)), ConvLSTM2D[Double](
      3,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 12, 5, 5))).asInstanceOf[Array[Recurrent[Double]]]

    val decoderCells = Array(ConvLSTM2D[Double](
      7,
      nbKernal,
      inputShape = Shape(input.size())), ConvLSTM2D[Double](
      12,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 7, 5, 5)), ConvLSTM2D[Double](
      3,
      nbKernal,
      inputShape = Shape(batchSize, seqLength, 12, 5, 5))).asInstanceOf[Array[Recurrent[Double]]]

    val model = Seq2seq(encoderCells, decoderCells)
    val output = model.forward(T(input, input)).toTensor

    val input2 = input.clone()
    val expect = output.clone()
    val serFile = java.io.File.createTempFile("UnitTest", "AnalyticsZooSpecBase")
    model.saveModel(serFile.getAbsolutePath, overWrite = true)
    val loadModel = ZooModel.loadModel(serFile.getAbsolutePath)
    val output2 = loadModel.forward(T(input2, input2)).toTensor
    expect.almostEqual(output2, 1e-6)
  }
}



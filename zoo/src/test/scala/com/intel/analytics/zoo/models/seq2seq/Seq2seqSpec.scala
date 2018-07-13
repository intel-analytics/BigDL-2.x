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

import com.intel.analytics.bigdl.nn.ConvLSTMPeephole
import com.intel.analytics.bigdl.nn.abstractnn.{TensorModule}
import com.intel.analytics.bigdl.tensor.{Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import com.intel.analytics.bigdl.nn._

class Seq2seqSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "A Seq2seq" should "work with PassThroughBridge" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val kernalW = 3
    val kernalH = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand

    val encoderCells = Array(ConvLSTMPeephole[Double](
      3,
      7,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      7,
      12,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      12,
      3,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val decoderCells = Array(ConvLSTMPeephole[Double](
      3,
      7,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      7,
      12,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      12,
      3,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val model = Seq2seq(encoderCells, decoderCells)

    model.parameters()
    model.getParametersTable()

    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    model.setLoop(seqLength)

    for (i <- 0 until 3) {
      val output = model.forward(T(input, input.select(2, seqLength))).toTensor
      model.backward(T(input, input.select(2, seqLength)), gradOutput)
    }
  }

  "A Seq2seq" should "work with ZeroBridge" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val kernalW = 3
    val kernalH = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand

    val encoderCells = Array(ConvLSTMPeephole[Double](
      3,
      7,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      7,
      12,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      12,
      3,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val decoderCells = Array(ConvLSTMPeephole[Double](
      3,
      7,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      7,
      12,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      12,
      3,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val model = Seq2seq(encoderCells, decoderCells, bridges = new ZeroBridge())

    model.parameters()
    model.getParametersTable()
    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    model.setLoop(seqLength)

    for (i <- 0 until 3) {
      val output = model.forward(T(input, input.select(2, seqLength))).toTensor
      model.backward(T(input, input.select(2, seqLength)), gradOutput)
    }
  }

  "A Seq2seq" should "work with InitialStateBridge" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val kernalW = 3
    val kernalH = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, 3, 5, 5).rand

    val encoderCells = Array(ConvLSTMPeephole[Double](
      3,
      7,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      7,
      12,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      12,
      3,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val decoderCells = Array(ConvLSTMPeephole[Double](
      3,
      14,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      14,
      25,
      kernalW, kernalH,
      1), ConvLSTMPeephole[Double](
      25,
      3,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val activations = Array(
      Array(SpatialConvolution[Double](7, 14, kernalW, kernalW, 1, 1, kernalW/2, kernalW/2),
      SpatialConvolution[Double](7, 14, kernalH, kernalH, 1, 1, kernalH/2, kernalH/2)),
      Array(SpatialConvolution[Double](12, 25, kernalW, kernalW, 1, 1, kernalW/2, kernalW/2),
        SpatialConvolution[Double](12, 25, kernalH, kernalH, 1, 1, kernalH/2, kernalH/2)), null
    ).asInstanceOf[Array[Array[TensorModule[Double]]]]
    val model = Seq2seq(encoderCells, decoderCells,
      bridges = new InitialStateBridge[Double](activations))
    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    model.setLoop(seqLength)

    for (i <- 0 until 3) {
      model.forward(T(input, input.select(2, seqLength))).toTensor
      model.backward(T(input, input.select(2, seqLength)), gradOutput)
    }
  }

  "A Seq2seq" should "work with InitialStateBridge2" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val kernalW = 3
    val kernalH = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, 3, 5, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, 5, 5, 5, 5).rand

    val encoderCells = Array(ConvLSTMPeephole3D[Double](
      3,
      7,
      kernalW, kernalH,
      1), ConvLSTMPeephole3D[Double](
      7,
      7,
      kernalW, kernalH,
      1), ConvLSTMPeephole3D[Double](
      7,
      7,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val decoderCells = Array(ConvLSTMPeephole3D[Double](
      5,
      5,
      kernalW, kernalH,
      1), ConvLSTMPeephole3D[Double](
      5,
      5,
      kernalW, kernalH,
      1), ConvLSTMPeephole3D[Double](
      5,
      5,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val activations = Array(
      Array(VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1),
        VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1)),
      Array(VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1),
        VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1)),
      Array(VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1),
        VolumetricConvolution[Double](7, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    ).asInstanceOf[Array[Array[TensorModule[Double]]]]

    val preDecoder = Sequential().add(Contiguous())
      .add(VolumetricConvolution[Double](3, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    val model = Seq2seq(encoderCells, decoderCells, preDecoder = preDecoder,
      bridges = new InitialStateBridge[Double](activations))
    model.setLoop(seqLength)

    for (i <- 0 until 3) {
      model.forward(T(input, input.select(2, seqLength)))
      model.backward(T(input, input.select(2, seqLength)), gradOutput)
    }
  }

  "A Seq2seq" should "work with single cell" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val hiddenSize = 7
    val inputSize = 7
    val kernalW = 3
    val kernalH = 3
    val seqLength = 5
    val seed = 100
    val batchSize = 4

    RNG.setSeed(seed)
    val input = Tensor[Double](batchSize, seqLength, inputSize, 5, 5).rand
    val gradOutput = Tensor[Double](batchSize, seqLength, hiddenSize, 5, 5).rand

    val encoderCells = Array(ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val decoderCells = Array(ConvLSTMPeephole[Double](
      inputSize,
      hiddenSize,
      kernalW, kernalH,
      1)).asInstanceOf[Array[Cell[Double]]]

    val model = Seq2seq(encoderCells, decoderCells)

    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    model.setLoop(seqLength)

    for (i <- 0 until 3) {
      val output = model.forward(T(input, input.select(2, seqLength))).toTensor
      model.backward(T(input, input.select(2, seqLength)), gradOutput)
    }
  }

  "A Seq2seq" should "work with getParameters" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val inputSize = 2
    val hiddenSize = 16
    val outputSize = 1
    val seed = 100

    RNG.setSeed(seed)

    val encoderCells = Array(ConvLSTMPeephole3D[Double](
      inputSize, hiddenSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]

    val decoderCells = Array(ConvLSTMPeephole3D[Double](
      outputSize, outputSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]

    val activations = Array(
      Array(VolumetricConvolution[Double](hiddenSize, outputSize, 3, 3, 3, 1, 1, 1, -1, -1, -1)),
      Array(VolumetricConvolution[Double](hiddenSize, outputSize, 3, 3, 3, 1, 1, 1, -1, -1, -1))
    ).asInstanceOf[Array[Array[TensorModule[Double]]]]

    val _bridges = new InitialStateBridge[Double](activations)
    val _preDecoder = Sequential().add(Contiguous())
    .add(VolumetricConvolution(inputSize, outputSize, 3, 3, 3, 1, 1, 1, -1, -1, -1))

    val model = Seq2seq(encoderCells, decoderCells, bridges = _bridges, preDecoder = _preDecoder)

    require(model.getParametersTable().length() == 19)
    require(model.parameters()._1.length == 30)
  }

  "A Seq2seq" should "work with getParameters 2" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val inputSize = 2
    val hiddenSize = 16
    val outputSize = 1
    val seed = 100

    RNG.setSeed(seed)

    val encoderCells = Array(ConvLSTMPeephole3D[Double](
      inputSize, hiddenSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]

    val decoderCells = Array(ConvLSTMPeephole3D[Double](
      outputSize, outputSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]

    val model = Seq2seq(encoderCells, decoderCells)

    val encoderCells2 = Array(ConvLSTMPeephole3D[Double](
      inputSize, hiddenSize, 3, 3, withPeephole = false),
      ConvLSTMPeephole3D[Double](
        inputSize, hiddenSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]

    val decoderCells2 = Array(ConvLSTMPeephole3D[Double](
      inputSize, hiddenSize, 3, 3, withPeephole = false),
      ConvLSTMPeephole3D[Double](
        inputSize, hiddenSize, 3, 3, withPeephole = false)).asInstanceOf[Array[Cell[Double]]]

    val model2 = Seq2seq(encoderCells2, decoderCells2)
    require(2 * model.getParametersTable().length() == model2.getParametersTable().length())
  }

//  "A Seq2seq" should "work with stop sign" in {
//    import com.intel.analytics.bigdl.numeric.NumericDouble
//    val hiddenSize = 7
//    val inputSize = 7
//    val kernalW = 3
//    val kernalH = 3
//    val seqLength = 5
//    val seed = 100
//    val batchSize = 4
//
//    RNG.setSeed(seed)
//    val input = Tensor[Double](batchSize, seqLength, inputSize, 5, 5)
//
//    val encoderCells = Array(ConvLSTMPeephole[Double](
//      inputSize,
//      hiddenSize,
//      kernalW, kernalH,
//      1)).asInstanceOf[Array[Cell[Double]]]
//
//    val decoderCells = Array(ConvLSTMPeephole[Double](
//      inputSize,
//      hiddenSize,
//      kernalW, kernalH,
//      1)).asInstanceOf[Array[Cell[Double]]]
//
//    val model = Seq2seq(encoderCells, decoderCells)
//    def stop(x: Tensor[Double]): Boolean = {
//      x.almostEqual(Tensor[Double](batchSize, hiddenSize, 5, 5), 1e-6)
//    }
////    model.setLoop(seqLength, stopSign = stop)
//    model.setLoop(seqLength)
//    var output = model.forward(T(input, input.select(2, seqLength))).toTensor
//    require(output.size(2) == seqLength)
//
//    model.parameters()._1.foreach(_.fill(0.0))
//    output = model.forward(T(input, input.select(2, seqLength))).toTensor
//    require(output.size(2) == 1)
//  }

//  "A Seq2seq" should "work with loop func" in {
//    import com.intel.analytics.bigdl.numeric.NumericDouble
//    val hiddenSize = 7
//    val inputSize = 7
//    val kernalW = 3
//    val kernalH = 3
//    val seqLength = 5
//    val seed = 100
//    val batchSize = 4
//
//    RNG.setSeed(seed)
//    val input = Tensor[Double](batchSize, seqLength, inputSize, 5, 5)
//
//    val encoderCells = Array(ConvLSTMPeephole[Double](
//      inputSize,
//      hiddenSize,
//      kernalW, kernalH,
//      1)).asInstanceOf[Array[Cell[Double]]]
//
//    val decoderCells = Array(ConvLSTMPeephole[Double](
//      inputSize,
//      hiddenSize,
//      kernalW, kernalH,
//      1)).asInstanceOf[Array[Cell[Double]]]
//
//    val model = Seq2seq(encoderCells, decoderCells)
////    model.setLoop(seqLength, null, input => input.add(1))
//    model.setLoop(seqLength)
//    val output = model.forward(T(input, input.select(2, seqLength))).toTensor
//    require(output.size(2) == seqLength)
//  }
}

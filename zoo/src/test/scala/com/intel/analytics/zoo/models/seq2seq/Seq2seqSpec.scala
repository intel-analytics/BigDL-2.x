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

    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    model.setLoopPreOutput(seqLength)

    for (i <- 0 until 3) {
      val output = model.forward(input).toTensor
      model.backward(input, gradOutput)
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

    for (i <- 0 until 3) {
      model.forward(T(input, input)).toTensor
      model.backward(T(input, input), gradOutput)
    }

    model.setLoopPreOutput(seqLength)

    for (i <- 0 until 3) {
      val output = model.forward(input).toTensor
      model.backward(input, gradOutput)
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

    model.setLoopPreOutput(seqLength)

    for (i <- 0 until 3) {
      model.forward(input).toTensor
      model.backward(input, gradOutput)
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
    model.setLoopPreOutput(seqLength)

    for (i <- 0 until 3) {
      model.forward(input)
      model.backward(input, gradOutput)
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

    model.setLoopPreOutput(seqLength)

    for (i <- 0 until 3) {
      val output = model.forward(input).toTensor
      model.backward(input, gradOutput)
    }
  }
}

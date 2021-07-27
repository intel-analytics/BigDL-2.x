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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, SingleShape, T, Table}
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class Seq2seqSpec extends FlatSpec with Matchers with BeforeAndAfter {

  "Seq2seq model with lstm" should "be able to work" in {
    val inputSize = 3
    val hiddenSize = 6
    val batchSize = 2
    val seqLen = 2
    val numLayer = 1
    val encoder = RNNEncoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))
    val decoder = RNNDecoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))

    val input = Tensor.ones[Float](batchSize, seqLen)
    val input2 = Tensor[Float](batchSize, seqLen)

    val gradOutput = Tensor[Float](batchSize, seqLen, hiddenSize).rand()
    val model = Seq2seq[Float](encoder, decoder,
      SingleShape(List(seqLen)), SingleShape(List(seqLen)))
    model.forward(T(input, input2))
    model.backward(T(input, input2), gradOutput)

    val encoder2 = RNNEncoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))
    val decoder2 = RNNDecoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))
    val bridge = Bridge[Float]("dense", hiddenSize)
    val model2 = Seq2seq[Float](encoder2, decoder2,
      SingleShape(List(-1)), SingleShape(List(-1)), bridge
      )
    model2.forward(T(input, input2))
    model2.backward(T(input, input2), gradOutput)

    val sent1 = Tensor(Array[Float](25f, 39f, 99f, 123f), Array(1, 2, 2))
    val sent2 = Tensor(Array[Float](45f, 60f), Array(1, 2))
    val encoder3 = RNNEncoder[Float]("lstm", numLayer, 2)
    val decoder3 = RNNDecoder[Float]("lstm", numLayer, 2)
    val model3 = Seq2seq[Float](encoder3, decoder3,
      SingleShape(List(2, 2)), SingleShape(List(2, 2)))

    val result = model3.infer(sent1, sent2, maxSeqLen = 3).toTensor[Float]
  }

  "Seq2seq model with customized rnn" should "be able to work" in {
    val inputSize = 3
    val hiddenSize = 6
    val batchSize = 2
    val seqLen = 2

    val encoderRNN = Array(LSTM[Float](4, returnSequences = true),
      LSTM[Float](5, returnSequences = true),
      LSTM[Float](hiddenSize, returnSequences = true))
      .asInstanceOf[Array[Recurrent[Float]]]
    val decoderRNN = Array(LSTM[Float](4, returnSequences = true),
      LSTM[Float](5, returnSequences = true),
      LSTM[Float](hiddenSize, returnSequences = true))
      .asInstanceOf[Array[Recurrent[Float]]]

    val encoder = RNNEncoder[Float](encoderRNN, Embedding[Float](10, inputSize), null)
    val decoder = RNNDecoder[Float](decoderRNN, Embedding[Float](10, inputSize), null)

    val input = Tensor.ones[Float](batchSize, seqLen)
    val input2 = Tensor[Float](batchSize, seqLen)

    val gradOutput = Tensor[Float](batchSize, seqLen, hiddenSize).rand()
    val model = Seq2seq[Float](encoder, decoder,
      SingleShape(List(seqLen)), SingleShape(List(seqLen)))
    model.forward(T(input, input2))
    model.backward(T(input, input2), gradOutput)
  }

  "Seq2seq model with lstm" should "be able to work with different" +
    "encoder/decoder hiddensize" in {
    val inputSize = 3
    val encoderHiddenSize = 4
    val decoderHiddenSize = 6
    val batchSize = 1
    val seqLen = 2
    val numLayer = 1
    val encoder = RNNEncoder[Float]("lstm", numLayer, encoderHiddenSize,
      Embedding[Float](10, inputSize))
    val decoder = RNNDecoder[Float]("lstm", numLayer, decoderHiddenSize,
      Embedding[Float](10, decoderHiddenSize))

    val input = Tensor.ones[Float](batchSize, seqLen)
    val input2 = Tensor[Float](batchSize, seqLen)

    val gradOutput = Tensor[Float](batchSize, seqLen, decoderHiddenSize).rand()

    val bridge = Bridge[Float]("dense", decoderHiddenSize)
    val model2 = Seq2seq[Float](encoder, decoder,
      SingleShape(List(seqLen)), SingleShape(List(seqLen)),
      bridge)
    model2.forward(T(input, input2))
    model2.backward(T(input, input2), gradOutput)
  }

  "Seq2seq model with multiple lstm" should "be able to work" in {
    val inputSize = 3
    val hiddenSize = 5
    val batchSize = 1
    val seqLen = 2
    val numLayer = 3
    val encoder = RNNEncoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))
    val decoder = RNNDecoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))

    val input = Tensor.ones[Float](batchSize, seqLen)
    val input2 = Tensor[Float](batchSize, seqLen)

    val gradOutput = Tensor[Float](batchSize, seqLen, hiddenSize).rand()
    val model = Seq2seq[Float](encoder, decoder,
      SingleShape(List(seqLen)), SingleShape(List(seqLen)))
    model.forward(T(input, input2))
    model.backward(T(input, input2), gradOutput)

    val encoder2 = RNNEncoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))
    val decoder2 = RNNDecoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))
    val bridge = Bridge[Float]("dense", hiddenSize)
    val model2 = Seq2seq[Float](encoder2, decoder2,
      SingleShape(List(seqLen)), SingleShape(List(seqLen)), bridge)
    model2.forward(T(input, input2))
    model2.backward(T(input, input2), gradOutput)

    val encoder3 = RNNEncoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))
    val decoder3 = RNNDecoder[Float]("lstm", numLayer, hiddenSize, Embedding[Float](10, inputSize))
    val model3 = Seq2seq[Float](encoder3, decoder3,
      SingleShape(List(seqLen)), SingleShape(List(seqLen)),
      bridge = new Bridge[Float](Dense[Float](hiddenSize * numLayer * 2)
        .asInstanceOf[KerasLayer[Tensor[Float], Tensor[Float], Float]]))
    model3.forward(T(input, input2))
    model3.backward(T(input, input2), gradOutput)
  }

  "Seq2seq model with multiple lstm" should "be able to work with different" +
    "encoder/decoder hiddensize" in {
    val inputSize = 3
    val encoderHiddenSize = 4
    val decoderHiddenSize = 5
    val batchSize = 1
    val seqLen = 2
    val numLayer = 3
    val encoder = RNNEncoder[Float]("lstm", numLayer, encoderHiddenSize,
      Embedding[Float](10, inputSize))
    val decoder = RNNDecoder[Float]("lstm", numLayer, decoderHiddenSize,
      Embedding[Float](10, decoderHiddenSize))

    val input = Tensor.ones[Float](batchSize, seqLen)
    val input2 = Tensor[Float](batchSize, seqLen)

    val gradOutput = Tensor[Float](batchSize, seqLen, decoderHiddenSize).rand()

    val bridge = Bridge[Float]("densenonlinear", decoderHiddenSize)
    val model = Seq2seq[Float](encoder, decoder,
      SingleShape(List(seqLen)), SingleShape(List(seqLen)), bridge)
    model.forward(T(input, input2))
    model.backward(T(input, input2), gradOutput)
  }

  "Seq2seq model with multiple simple rnn" should "be able to work" in {
    val inputSize = 10
    val hiddenSize = 3
    val batchSize = 2
    val seqLen = 4
    val numLayer = 2
    val encoder = RNNEncoder[Float]("SimpleRNN", numLayer, hiddenSize)
    val decoder = RNNDecoder[Float]("SimpleRNN", numLayer, hiddenSize)

    val input = Tensor[Float](batchSize, seqLen, inputSize).rand()
    val input2 = Tensor[Float](batchSize, seqLen, inputSize).rand()

    val gradOutput = Tensor[Float](batchSize, seqLen, hiddenSize).rand()
    val model = Seq2seq[Float](encoder, decoder,
      SingleShape(List(seqLen, inputSize)), SingleShape(List(seqLen, inputSize)),
      Bridge[Float]("dense", hiddenSize))

    val output = model.forward(T(input, input2))
    val t = model.backward(T(input, input2), gradOutput)

    val encoder2 = RNNEncoder[Float]("SimpleRNN", numLayer, hiddenSize)
    val decoder2 = RNNDecoder[Float]("SimpleRNN", numLayer, hiddenSize)

    val model2 = Seq2seq[Float](encoder2, decoder2,
      SingleShape(List(seqLen, inputSize)), SingleShape(List(seqLen, inputSize)),
      bridge = new Bridge[Float](Dense[Float](hiddenSize * numLayer)
        .asInstanceOf[KerasLayer[Tensor[Float], Tensor[Float], Float]]))

    model2.forward(T(input, input2))
    model2.backward(T(input, input2), gradOutput)
  }

  "Seq2seq model with simple rnn" should "generate correct result" in {
    val inputSize = 10
    val hiddenSize = 3
    val batchSize = 2
    val seqLen = 4
    val numLayer = 1
    val encoder = RNNEncoder[Float]("SimpleRNN", numLayer, hiddenSize)
    val decoder = RNNDecoder[Float]("SimpleRNN", numLayer, hiddenSize)

    val input = Tensor[Float](batchSize, seqLen, inputSize)
    val input2 = Tensor[Float](batchSize, seqLen-1, inputSize)

    val gradOutput = Tensor[Float](Array[Float](0.5535f, 0.4117f, 0.3510f,
        0.3881f, 0.5073f, 0.4701f,
        0.3155f, 0.9211f, 0.6948f,

        0.8196f, 0.9297f, 0.4505f,
        0.6202f, 0.6401f, 0.0459f,
        0.4751f, 0.1985f, 0.1941f),
      Array(batchSize, seqLen - 1, hiddenSize))
    val model = Seq2seq[Float](encoder, decoder,
      SingleShape(List(seqLen, inputSize)), SingleShape(List(seqLen-1, inputSize)))

    val w3 = Tensor[Float](Array[Float](-0.5080f, -0.2488f, -0.3456f, 0.0016f,
      -0.2148f, -0.0400f, -0.3912f, -0.3963f, -0.3368f, -0.1976f,
      -0.4557f, 0.4841f, -0.1146f, 0.4968f, 0.1799f, -0.4889f, 0.3995f, -0.1589f,
      -0.2213f, -0.4792f, -0.5740f, 0.1652f, -0.1261f, 0.2248f, -0.4738f, 0.4286f,
      -0.4238f, -0.0997f, 0.1206f, 0.2981f),
      Array(3, 10))
    val w4 = Tensor[Float](Array[Float](0.4661f, 0.5259f, -0.4578f,
      0.1453f, -0.2483f, -0.0633f,
      -0.4321f, 0.5259f, -0.4237f), Array(3, 3))
    val w5 = Tensor[Float](Array[Float](0.3086f, 0.2029f, 0.1876f), Array(3))
    val w0 = Tensor[Float](Array[Float](0.5556f, -0.4765f, -0.5727f, -0.4517f, -0.3884f,
      0.2339f, 0.2067f, 0.4797f, -0.2982f, -0.3936f, 0.3063f, -0.2334f,
      0.3504f, -0.1370f, 0.3303f, -0.4486f, -0.2914f, 0.1760f, 0.1221f, -0.1472f,
      0.3441f, 0.3925f, -0.4187f, -0.3082f, 0.5287f, -0.1948f, -0.2047f, -0.5586f,
      -0.3306f, 0.1442f), Array(3, 10))
    val w1 = Tensor[Float](Array[Float](-0.0762f, -0.4191f, 0.0135f,
      -0.3944f, -0.4898f, -0.3179f,
      -0.5053f, -0.3676f, 0.5771f), Array(3, 3))
    val w2 = Tensor[Float](Array[Float](0.1090f, 0.1779f, -0.5385f), Array(3))
    val w = model.parameters()._1
    w(3).set(w3)
    w(4).set(w4)
    w(5).set(w5)
    w(0).set(w0)
    w(1).set(w1)
    w(2).set(w2)

    val output = model.forward(T(input, input2))
    val expectO = Tensor[Float](Array[Float](0.6669f, 0.1809f, 0.5919f,
      0.6669f, 0.1809f, 0.5919f,
      0.4166f, 0.2141f, -0.2508f,
      0.4166f, 0.2141f, -0.2508f,

      0.6232f, 0.2224f, 0.2227f,
      0.6232f, 0.2224f, 0.2227f), Array(seqLen-1, batchSize, hiddenSize))
    assert(output.transpose(1, 2).almostEqual(expectO, 1e-4) == true)
    val encoderO = model.asInstanceOf[Seq2seq[Float]].encoder.output.toTable
    val memoryBank = encoderO[Tensor[Float]](1).transpose(1, 2)
    val expectMemoryBank = Tensor[Float](Array[Float](0.1086f, 0.1761f, -0.4918f,
      0.1086f, 0.1761f, -0.4918f,

      0.0203f, 0.2024f, -0.7361f,
      0.0203f, 0.2024f, -0.7361f,

      0.0127f, 0.2957f, -0.7810f,
      0.0127f, 0.2957f, -0.7810f,

      -0.0264f, 0.2695f, -0.8021f,
      -0.0264f, 0.2695f, -0.8021f), Array(seqLen, batchSize, hiddenSize))
    assert(memoryBank.almostEqual(expectMemoryBank, 1e-4) == true)

    val t = model.backward(T(input, input2), gradOutput)
    val gradients = model.parameters()._2
    val expect_g0 = Tensor[Float](3, inputSize)
    val expect_g1 = Tensor[Float](Array[Float](0.0259f, 0.1994f, -0.5182f,
      0.0210f, 0.0824f, -0.1596f,
      -0.0031f, -0.0908f, 0.2592f), Array(3, hiddenSize))
    val expect_g2 = Tensor[Float](Array[Float](0.6213f, 0.0846f, -0.3706f), Array(3))
    val expect_g3 = Tensor[Float](3, inputSize)
    val expect_g4 = Tensor[Float](Array[Float](0.7355f, 0.5617f, -0.5289f,
      1.4155f, 0.8520f, -0.4186f,
      0.2639f, 0.2236f, -0.4776f), Array(3, hiddenSize))
    val expect_g5 = Tensor[Float](Array[Float](2.4615f, 3.8764f, 0.9632f), Array(3))
    assert(gradients(0).almostEqual(expect_g0, 1e-4) == true)
    assert(gradients(1).almostEqual(expect_g1, 1e-4) == true)
    assert(gradients(2).almostEqual(expect_g2, 1e-4) == true)
    assert(gradients(3).almostEqual(expect_g3, 1e-4) == true)
    assert(gradients(4).almostEqual(expect_g4, 1e-3) == true)
    assert(gradients(5).almostEqual(expect_g5, 1e-4) == true)

    val encoder2 = RNNEncoder[Float]("SimpleRNN", numLayer, hiddenSize)
    val decoder2 = RNNDecoder[Float]("SimpleRNN", numLayer, hiddenSize)
    val model2 = Seq2seq[Float](encoder2, decoder2,
      SingleShape(List(seqLen, inputSize)), SingleShape(List(seqLen, inputSize)),
      bridge = new Bridge[Float](Identity[Float]()
        .asInstanceOf[KerasLayer[Tensor[Float], Tensor[Float], Float]]))
    val w_2 = model2.parameters()._1
    w_2(3).set(w3)
    w_2(4).set(w4)
    w_2(5).set(w5)
    w_2(0).set(w0)
    w_2(1).set(w1)
    w_2(2).set(w2)
    val output2 = model2.forward(T(input, input2))
    model2.backward(T(input, input2), gradOutput)
    val gradients2 = model2.parameters()._2
    assert(output.almostEqual(output2, 1e-8) == true)
    for (i <- 0 until gradients.size) {
      assert(gradients(i).almostEqual(gradients2(i), 1e-8) == true)
    }
  }

  "Simple rnn" should "generate correct result" in {
    val inputSize = 10
    val hiddenSize = 3
    val batchSize = 3
    val seqLen = 2
    val seq = Sequential[Float]()
    val rnn = SimpleRNN[Float](hiddenSize, inputShape = SingleShape(List(seqLen, inputSize)))
    seq.add(rnn)

    val input = Tensor[Float](Array[Float](0.2814f, 0.7886f, 0.5895f, 0.7539f, 0.1952f,
      0.0050f, 0.3068f, 0.1165f, 0.9103f, 0.6440f, 0.7071f, 0.6581f, 0.4913f, 0.8913f,
      0.1447f, 0.5315f, 0.1587f, 0.6542f, 0.3278f, 0.6532f,

      0.3958f, 0.9147f, 0.2036f, 0.2018f, 0.2018f, 0.9497f, 0.6666f, 0.9811f,
      0.0874f, 0.0041f,
      0.1088f, 0.1637f, 0.7025f, 0.6790f, 0.9155f, 0.2418f, 0.1591f, 0.7653f,
      0.2979f, 0.8035f,

      0.3813f, 0.7860f, 0.1115f, 0.2477f, 0.6524f, 0.6057f, 0.3725f, 0.7980f,
      0.8399f, 0.1374f,
      0.2331f, 0.9578f, 0.3313f, 0.3227f, 0.0162f, 0.2137f, 0.6249f, 0.4340f,
      0.1371f, 0.5117f), Array(batchSize, seqLen, inputSize))
    val gradOutput = Tensor[Float](Array[Float](0.1585f, 0.0758f, 0.2247f,
      0.0624f, 0.1816f, 0.9998f,
      0.5944f, 0.6541f, 0.0337f), Array(batchSize, 1, hiddenSize))

    val w = seq.parameters()._1
    w(0).set(Tensor[Float](Array[Float](0.4414f, 0.4792f, -0.1353f, 0.5304f, -0.1265f,
      0.1165f, -0.2811f, 0.3391f, 0.5090f, -0.4236f, 0.5018f, 0.1081f, 0.4266f,
      0.0782f, 0.2784f, -0.0815f, 0.4451f, 0.0853f, -0.2695f, 0.1472f,
      -0.2660f, -0.0677f, -0.2345f, 0.3830f, -0.4557f, -0.2662f, -0.1630f, -0.3471f,
      0.0545f, -0.5702f),
      Array(3, 10)))
    w(1).set(Tensor[Float](Array[Float](0.5214f, -0.4904f, 0.4457f,
      0.0961f, -0.1875f, 0.3568f,
      0.0900f, 0.4665f, 0.0631f), Array(3, 3)))
    w(2).set(Tensor[Float](Array[Float](-0.1821f, 0.1551f, -0.1566f), Array(3)))

    val output = seq.forward(input).toTensor[Float]
    val expectO = Tensor[Float](Array[Float](0.6281f, 0.6263f, -0.5795f,
      -0.2112f, 0.5615f, -0.7511f,
      0.1154f, 0.5124f, -0.5010f), Array(batchSize, 1, hiddenSize))
    assert(output.almostEqual(expectO, 1e-4) == true)
    val gradInput = seq.backward(input, gradOutput)
    val gradients = seq.parameters()._2
    val g0 = Tensor[Float](Array[Float](0.2974f, 0.8258f, 0.3330f, 0.3917f,
      0.1920f, 0.3254f, 0.4901f, 0.5321f, 0.2947f, 0.4570f,
      0.0972f, 0.3953f, 0.2626f, 0.2420f, -0.0144f, 0.0803f, 0.2896f, 0.2136f,
      -0.0728f, 0.3490f,
      0.2488f, 0.3925f, 0.4412f, 0.5204f, 0.5441f, 0.3237f, 0.2083f, 0.6135f,
      0.3656f, 0.5135f), Array(batchSize, inputSize))
    val g1 = Tensor[Float](Array[Float](0.5478f, 0.4415f, -0.5645f,
      0.4788f, 0.3961f, -0.5058f,
      0.4002f, 0.4189f, -0.4501f), Array(batchSize, hiddenSize))
    val g2 = Tensor[Float](Array[Float](0.9774f, 0.4912f, 0.8570f), Array(batchSize))
    assert(gradients(0).almostEqual(g0, 1e-3) == true)
    assert(gradients(1).almostEqual(g1, 1e-4) == true)
    assert(gradients(2).almostEqual(g2, 1e-3) == true)
  }
}

class RNNEncoderSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val cell = LSTM[Float](3).asInstanceOf[Recurrent[Float]]
    val layer = RNNEncoder[Float](Array(cell), Embedding[Float](10, 4), Shape(2))
    layer.build(Shape(1, 2))
    val w = layer.parameters()._1
    w.foreach(_.fill(100.0f))
    val input = Tensor[Float](1, 2).rand()
    runSerializationTest(layer, input)
  }
}

class RNNDecoderSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val shape1 = SingleShape(List(6))
    val shape2 = SingleShape(List(6))
    val mul = MultiShape(List(MultiShape(List(shape1, shape2))))

    val shape3 = SingleShape(List(1, 6))
    val shape4 = SingleShape(List(1, 6))
    val mul2 = MultiShape(List(MultiShape(List(shape3, shape4))))
    val shape = MultiShape(List(Shape(2), mul))
    val layer = RNNDecoder[Float]("lstm", 1, 6, inputShape = shape,
      embedding = Embedding[Float](100, 6))

    layer.build(MultiShape(List(Shape(1, 2), mul2)))
    val w = layer.parameters()._1
    w.foreach(_.fill(50.0f))
    val states = T(T(Tensor[Float](1, 6).rand(), Tensor[Float](1, 6).rand()))
    val input = T(Tensor[Float](1, 2).rand(), states)
    runSerializationTest(layer, input)
  }
}

class Seq2seqSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val encoder = RNNEncoder[Float]("lstm", 1, 2)
    val decoder = RNNDecoder[Float]("lstm", 1, 2)

    val input = Tensor.ones[Float](1, 2, 2)
    val input2 = Tensor[Float](1, 2, 2)
    val model = Seq2seq[Float](encoder, decoder,
      SingleShape(List(2, 2)), SingleShape(List(2, 2)))
    val w = model.parameters()._1
    w.foreach(_.fill(150.0f))

    ZooSpecHelper.testZooModelLoadSave2(
      model.asInstanceOf[ZooModel[Table, Tensor[Float], Float]],
      T(input, input2), Seq2seq.loadModel[Float])
  }
}

class BridgeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val shape3 = SingleShape(List(2, 2))
    val shape4 = SingleShape(List(2, 2))

    val mul2 = MultiShape(List(MultiShape(List(shape3, shape4))))
    val layer = Bridge[Float]("dense", 2)
    layer.build(mul2)
    val input = T(T(Tensor[Float](2, 2).rand(), Tensor[Float](2, 2).rand()))
    runSerializationTest(layer, input)
  }
}

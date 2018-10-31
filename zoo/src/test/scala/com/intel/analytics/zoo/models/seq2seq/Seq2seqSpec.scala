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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, SingleShape, T}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Embedding, LSTM, Recurrent}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class Seq2seqSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test ObjectDetector").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "Seq2seq model with 1 rnn" should "be able to work" in {
    val inputSize = 3
    val hiddenSize = 6
    val batchSize = 1
    val seqLen = 2
    val encoder = Encoder[Float](Array(LSTM[Float](hiddenSize, returnSequences = true)), Embedding(10, inputSize))
    val decoder = Decoder[Float](Array(LSTM[Float](hiddenSize, returnSequences = true)), Embedding(10, inputSize))

    val input = Tensor.ones[Float](batchSize, seqLen)
    val input2 = Tensor[Float](batchSize, seqLen)

    val gradOutput = Tensor[Float](batchSize, seqLen, hiddenSize).rand()
    val model = Seq2seq[Float](encoder, decoder, Shape(List(SingleShape(List(seqLen)), SingleShape(List(seqLen)))))
    model.forward(T(input, input2))
    model.backward(T(input, input2), gradOutput)
  }

  "Seq2seq model with multiple rnns" should "be able to work" in {
    val inputSize = 3
    val hiddenSize = 5
    val batchSize = 1
    val seqLen = 2
    val encoder = Encoder[Float](Array(LSTM[Float](hiddenSize, returnSequences = true),
      LSTM[Float](hiddenSize, returnSequences = true),
      LSTM[Float](hiddenSize, returnSequences = true)), Embedding(10, inputSize))
    val decoder = Decoder[Float](Array(LSTM[Float](hiddenSize, returnSequences = true),
      LSTM[Float](hiddenSize, returnSequences = true),
      LSTM[Float](hiddenSize, returnSequences = true)), Embedding(10, inputSize))

    val input = Tensor.ones[Float](batchSize, seqLen)
    val input2 = Tensor[Float](batchSize, seqLen)

    val gradOutput = Tensor[Float](batchSize, seqLen, hiddenSize).rand()
    val model = Seq2seq[Float](encoder, decoder, Shape(List(SingleShape(List(seqLen)), SingleShape(List(seqLen)))))
    model.forward(T(input, input2))
    model.backward(T(input, input2), gradOutput)
  }
}

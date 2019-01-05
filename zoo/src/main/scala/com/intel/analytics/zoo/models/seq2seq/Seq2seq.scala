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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{PaddingParam, Sample}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.optim.{OptimMethod, ValidationMethod}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * [[Seq2seq]] A trainable interface for a simple, generic encoder + decoder model
 * @param encoder an encoder object
 * @param decoder a decoder object
 * @param inputShape shape of encoder input, for variable length, please input -1
 * @param outputShape shape of decoder input, for variable length, please input -1
 * @param bridge connect encoder and decoder
 */
class Seq2seq[T: ClassTag] (
  val encoder: Encoder[T],
  val decoder: Decoder[T],
  inputShape: Shape,
  outputShape: Shape,
  bridge: KerasLayer[Activity, Activity, T],
  generator: KerasLayer[Activity, Activity, T])
  (implicit ev: TensorNumeric[T]) extends ZooModel[Table, Tensor[T], T] {

  override def buildModel(): AbstractModule[Table, Tensor[T], T] = {
    val encoderInput = Input(inputShape)
    val decoderInput = Input(outputShape)

    val encoderOutput = encoder.inputs(encoderInput)

    // select table is 0 based
    val encoderFinalStates = SelectTable(1).inputs(encoderOutput)
    val decoderInitStates = if (bridge != null) {
      bridge.inputs(encoderFinalStates)
    }
    else encoderFinalStates

    val decoderOutput = decoder.inputs(Array(decoderInput, decoderInitStates))

    val output = if (generator != null) {
      generator.inputs(decoderOutput)
    }
    else decoderOutput

    Model(Array(encoderInput, decoderInput), output)
      .asInstanceOf[AbstractModule[Table, Tensor[T], T]]
  }

  def compile(
    optimizer: OptimMethod[T],
    loss: Criterion[T],
    metrics: List[ValidationMethod[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
model.asInstanceOf[KerasNet[T]].compile(optimizer, loss, metrics)
}

  def fit(
    x: RDD[Sample[T]],
    batchSize: Int = 32,
    nbEpoch: Int = 10,
    validationData: RDD[Sample[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
      model.asInstanceOf[KerasNet[T]].fit(x, batchSize, nbEpoch, validationData)
  }

  def infer(input: Tensor[T], startSign: Tensor[T], maxSeqLen: Int = 30,
            stopSign: Tensor[T] = null,
            buildOutput: KerasLayer[Tensor[T], Tensor[T], T] = null): Tensor[T] = {
    val sent1 = input
    val sent2 = startSign
    sent2.resize(Array(1, 1) ++ startSign.size())

    var curInput = sent2
    val sizes = curInput.size()
    val concat = Tensor[T](Array(sizes(0), maxSeqLen + 1) ++ sizes.drop(2))
    concat.narrow(Seq2seq.timeDim, 1, 1).copy(sent2)
    var break = false

    if (!buildOutput.isBuilt()) {
      buildOutput.build(generator.getOutputShape())
    }
    var j = 1
    // Iteratively output predicted words
    while (j <= maxSeqLen && !break) {
      val modelOutput = updateOutput(T(sent1, curInput)).toTensor[T]
      val generateOutput = if (buildOutput != null) buildOutput.forward(modelOutput)
      else modelOutput
      val predict = generateOutput.select(2, generateOutput.size(2))

      if (stopSign != null && predict.almostEqual(stopSign, 1e-8)) break = true
      j += 1
      concat.narrow(Seq2seq.timeDim, j, 1).copy(predict)
      curInput = concat.narrow(Seq2seq.timeDim, 1, j)
    }
    curInput
  }
}

object Seq2seq {
  val timeDim = 2
  /**
   * [[Seq2seq]] A trainable interface for a simple, generic encoder + decoder model
   * @param encoder a rnn encoder object
   * @param decoder a rnn decoder object
   * @param inputShape shape of encoder input, for variable length, please input -1
   * @param outputShape shape of decoder input, for variable length, please input -1
   * @param bridge connect encoder and decoder
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    encoder: RNNEncoder[T],
    decoder: RNNDecoder[T],
    inputShape: Shape,
    outputShape: Shape,
    bridge: KerasLayer[Activity, Activity, T] = null,
    generator: KerasLayer[Activity, Activity, T] = null
  )(implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    require(encoder.rnns.length == decoder.rnns.length, "rnn encoder and decoder should has" +
      " the same number of layers!")
    new Seq2seq[T](encoder, decoder, inputShape, outputShape,
      bridge, generator).build()
  }

  /**
   * Load an existing seq2seq model (with weights).
   *
   * @param path The path for the pre-defined model.
   *             Local file system, HDFS and Amazon S3 are supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx".
   *             Amazon S3 path should be like "s3a://bucket/xxx".
   * @param weightPath The path for pre-trained weights if any. Default is null.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def loadModel[T: ClassTag](
    path: String,
    weightPath: String = null)(implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[Seq2seq[T]]
  }
}

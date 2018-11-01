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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.layers.{SelectTable, Input}

import scala.reflect.ClassTag

/**
 * [[Seq2seq]] A trainable interface for a simple, generic encoder + decoder model
 * @param encoder an encoder object
 * @param decoder a decoder object
 * @param inputShape shape of input
 * @param bridge connect encoder and decoder
 */
class Seq2seq[T: ClassTag](
  encoder: Encoder[T],
  decoder: Decoder[T],
  inputShape: Shape,
  bridge: Bridge[T] = null)
  (implicit ev: TensorNumeric[T]) extends ZooModel[Table, Tensor[T], T] {

  override def buildModel(): AbstractModule[Table, Tensor[T], T] = {
    val encodeInputShape = inputShape.toMulti().head
    val decodeInputShape = inputShape.toMulti().last
    val encoderInput = Input(encodeInputShape)
    val decoderInput = Input(decodeInputShape)

    val encoderOutput = encoder.inputs(encoderInput)

    val encoderFinalStates = SelectTable(2).inputs(encoderOutput)
    val decoderInitStates = if (bridge != null) bridge.inputs(encoderFinalStates)
    else encoderFinalStates
    val decoderOutput = decoder.inputs(Array(decoderInput, decoderInitStates))

    Model(Array(encoderInput, decoderInput), decoderOutput)
      .asInstanceOf[AbstractModule[Table, Tensor[T], T]]
  }
}

object Seq2seq {
  /**
   * [[Seq2seq]] A trainable interface for a simple, generic encoder + decoder model
   * @param encoder an encoder object
   * @param decoder a decoder object
   * @param inputShape shape of input
   * @param bridge connect encoder and decoder
   */
  def apply[@specialized(Float, Double) T: ClassTag](
    encoder: Encoder[T],
    decoder: Decoder[T],
    inputShape: Shape,
    bridge: Bridge[T] = null
  )(implicit ev: TensorNumeric[T]): Seq2seq[T] = {
    new Seq2seq[T](encoder, decoder, inputShape, bridge).build()
  }
}
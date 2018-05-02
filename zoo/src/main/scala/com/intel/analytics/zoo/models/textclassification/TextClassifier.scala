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

package com.intel.analytics.zoo.models.textclassification

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

import scala.reflect.ClassTag

/**
 * The model used for text classification.
 *
 * @param classNum The number of text categories to be classified. Positive integer.
 * @param tokenLength The size of each word vector. Positive integer.
 * @param sequenceLength The length of a sequence. Positive integer. Default is 500.
 * @param encoder The encoder for input sequences. String. 'cnn' or 'lstm' or 'gru'.
 *                Default is 'cnn'.
 * @param encoderOutputDim The output dimension for the encoder. Positive integer. Default is 256.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class TextClassifier[T: ClassTag] private (
    val classNum: Int,
    val tokenLength: Int,
    val sequenceLength: Int = 500,
    val encoder: String = "cnn",
    val encoderOutputDim: Int = 256)(implicit ev: TensorNumeric[T])
  extends ZooModel[Tensor[T], Tensor[T], T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {
    val model = Sequential[T]()
    model.add(InputLayer(inputShape = Shape(sequenceLength, tokenLength)))
    if (encoder.toLowerCase() == "cnn") {
      model.add(Convolution1D(encoderOutputDim, 5, activation = "relu"))
      model.add(GlobalMaxPooling1D())
    }
    else if (encoder.toLowerCase() == "lstm") {
      model.add(LSTM(encoderOutputDim))
    }
    else if (encoder.toLowerCase() == "gru") {
      model.add(GRU(encoderOutputDim))
    }
    else {
      throw new IllegalArgumentException(s"Unsupported encoder: $encoder")
    }
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(Dense(classNum, activation = "softmax"))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object TextClassifier {
  /**
   * The factory method to create a TextClassifier instance.
   */
  def apply[@specialized(Float, Double) T: ClassTag](
      classNum: Int,
      tokenLength: Int,
      sequenceLength: Int = 500,
      encoder: String = "cnn",
      encoderOutputDim: Int = 256)(implicit ev: TensorNumeric[T]): TextClassifier[T] = {
    new TextClassifier[T](classNum, tokenLength, sequenceLength, encoder, encoderOutputDim).build()
  }

  /**
   * Load an existing TextClassifier model (with weights).
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
      weightPath: String = null)(implicit ev: TensorNumeric[T]): TextClassifier[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[TextClassifier[T]]
  }
}

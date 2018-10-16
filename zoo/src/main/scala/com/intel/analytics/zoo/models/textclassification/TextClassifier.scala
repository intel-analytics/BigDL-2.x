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

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.{OptimMethod, ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.feature.text.TextSet
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Sequential}

import scala.reflect.ClassTag

/**
 * The model used for text classification.
 */
class TextClassifier[T: ClassTag] private(
    val classNum: Int,
    val tokenLength: Int,
    val sequenceLength: Int = 500,
    val encoder: String = "cnn",
    val encoderOutputDim: Int = 256,
    val embedding: Embedding[T] = null)(implicit ev: TensorNumeric[T])
  extends ZooModel[Activity, Activity, T] {

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    val model = Sequential[T]()
    if (embedding != null) {
      model.add(embedding)
    }
    else {
      model.add(InputLayer(inputShape = Shape(sequenceLength, tokenLength)))
    }
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
      throw new IllegalArgumentException(s"Unsupported encoder for TextClassifier: $encoder")
    }
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(Dense(classNum, activation = "softmax"))
    model
  }

  def compile(
      optimizer: OptimMethod[T],
      loss: Criterion[T],
      metrics: List[ValidationMethod[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].compile(optimizer, loss, metrics)
  }

  def fit(
      x: TextSet,
      batchSize: Int,
      nbEpoch: Int,
      validationData: TextSet = null)(implicit ev: TensorNumeric[T]): Unit = {
    model.asInstanceOf[KerasNet[T]].fit(x, batchSize, nbEpoch, validationData)
  }

  def evaluate(
      x: TextSet,
      batchSize: Int)
    (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    model.asInstanceOf[KerasNet[T]].evaluate(x, batchSize)
  }

  def predict(
      x: TextSet,
      batchPerThread: Int): TextSet = {
    model.asInstanceOf[KerasNet[T]].predict(x, batchPerThread)
  }
}

object TextClassifier {
  /**
   * The factory method to create a TextClassifier instance with WordEmbedding as
   * its first layer.
   *
   * @param classNum The number of text categories to be classified. Positive integer.
   * @param embeddingFile The path to the embedding file.
   *                      Currently only the following GloVe files are supported:
   *                      "glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt"
   *                      "glove.6B.300d.txt", "glove.42B.300d.txt", "glove.840B.300d.txt".
   *                      You can download them from: https://nlp.stanford.edu/projects/glove/.
   * @param wordIndex Map of word (String) and its corresponding index (integer).
   *                  The index is supposed to start from 1 with 0 reserved for unknown words.
   *                  During the prediction, if you have words that are not in the wordIndex
   *                  for the training, you can map them to index 0.
   *                  Default is null. In this case, all the words in the embeddingFile will
   *                  be taken into account and you can call
   *                  WordEmbedding.getWordIndex(embeddingFile) to retrieve the map.
   * @param sequenceLength The length of a sequence. Positive integer. Default is 500.
   * @param encoder The encoder for input sequences. String. "cnn" or "lstm" or "gru" are supported.
   *                Default is "cnn".
   * @param encoderOutputDim The output dimension for the encoder. Positive integer. Default is 256.
   * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
   */
  def apply[@specialized(Float, Double) T: ClassTag](
      classNum: Int,
      embeddingFile: String,
      wordIndex: Map[String, Int] = null,
      sequenceLength: Int = 500,
      encoder: String = "cnn",
      encoderOutputDim: Int = 256)(implicit ev: TensorNumeric[T]): TextClassifier[T] = {
    val embedding = WordEmbedding(embeddingFile, wordIndex, inputLength = sequenceLength)
    new TextClassifier[T](classNum, embedding.outputDim, sequenceLength, encoder,
      encoderOutputDim, embedding).build()
  }

  /**
   * The factory method to create a TextClassifier instance that takes word vectors as input.
   */
  @deprecated("Instead of using 'tokenLength', please pass the arguments 'embeddingFile' " +
    "and 'wordIndex' to construct a TextClassifier with WordEmbedding as the first layer.")
  def apply[@specialized(Float, Double) T: ClassTag](
      classNum: Int,
      tokenLength: Int,
      sequenceLength: Int,
      encoder: String,
      encoderOutputDim: Int)(implicit ev: TensorNumeric[T]): TextClassifier[T] = {
    new TextClassifier[T](classNum, tokenLength, sequenceLength, encoder, encoderOutputDim).build()
  }

  /**
   * This factory method is mainly for Python use.
   * Pass in a model to build the TextClassifier.
   * Note that if you use this factory method, arguments such as classNum, tokenLength, etc
   * should match the model definition to eliminate ambiguity.
   */
  private[zoo] def apply[@specialized(Float, Double) T: ClassTag](
      classNum: Int,
      embedding: Embedding[T],
      sequenceLength: Int,
      encoder: String,
      encoderOutputDim: Int,
      model: AbstractModule[Activity, Activity, T])
    (implicit ev: TensorNumeric[T]): TextClassifier[T] = {
    new TextClassifier[T](classNum, embedding.outputDim, sequenceLength,
      encoder, encoderOutputDim, embedding).addModel(model)
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

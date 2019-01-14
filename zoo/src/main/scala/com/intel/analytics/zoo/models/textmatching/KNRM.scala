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

package com.intel.analytics.zoo.models.textmatching

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.models.common.ZooModel
import com.intel.analytics.zoo.pipeline.api.autograd.{Variable, AutoGrad => A}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Model

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Kernel-pooling Neural Ranking Model with RBF kernel.
 * https://arxiv.org/abs/1706.06613
 *
 * Input shape: (batch, text1Length + text2Length)
 * Every single input is expected to be the concatenation of text1 sequence and text2 sequence.
 * Output shape: (batch, 1)
 *
 * @param text1Length Sequence length of text1 (query).
 * @param text2Length Sequence length of text2 (doc).
 * @param vocabSize Integer. The inputDim of the embedding layer. Ought to be the total number
 *                  of words in the corpus +1, with index 0 reserved for unknown words.
 * @param embedSize Integer. The outputDim of the embedding layer. Default is 300.
 * @param embedWeights Tensor. Pre-trained word embedding weights if any. Default is null and in
 *                     this case, initial weights will be randomized.
 * @param trainEmbed Boolean. Whether to train the embedding layer or not. Default is true.
 * @param kernelNum Integer > 1. The number of kernels to use. Default is 21.
 * @param sigma Double. Defines the kernel width, or the range of its softTF count.
 *              Default is 0.1.
 * @param exactSigma Double. The sigma used for the kernel that harvests exact matches
 *                   in the case where RBF mu=1.0. Default is 0.001.
 * @param targetMode String. The target mode of the model. Either 'ranking' or 'classification'.
 *                   For ranking, the output will be the relevance score between text1 and text2 and
 *                   you are recommended to use 'rank_hinge' as loss for pairwise training.
 *                   For classification, the last layer will be sigmoid and the output will be the
 *                   probability between 0 and 1 indicating whether text1 is related to text2 and
 *                   you are recommended to use 'binary_crossentropy' as loss for binary
 *                   classification. Default mode is 'ranking'.
 */
class KNRM[T: ClassTag] private(
    override val text1Length: Int,
    val text2Length: Int,
    override val vocabSize: Int,
    override val embedSize: Int = 300,
    override val embedWeights: Tensor[T] = null,
    override val trainEmbed: Boolean = true,
    val kernelNum: Int = 21,
    val sigma: Double = 0.1,
    val exactSigma: Double = 0.001,
    override val targetMode: String = "ranking")(implicit ev: TensorNumeric[T])
  extends TextMatcher[T](text1Length, vocabSize, embedSize, embedWeights, trainEmbed, targetMode) {

  require(kernelNum > 1, s"kernelNum must be an integer greater than 1, but got $kernelNum")

  override def buildModel(): AbstractModule[Activity, Activity, T] = {
    // Remark: Share weights for embedding is not supported.
    // Thus here the model takes concatenated input and slice to split the input.
    val input = Input(inputShape = Shape(text1Length + text2Length))
    val embedding = Variable(Embedding(vocabSize, embedSize, weights = embedWeights,
      trainable = trainEmbed).inputs(input))
    val text1Embed = embedding.slice(1, 0, text1Length)
    val text2Embed = embedding.slice(1, text1Length, text2Length)
    val mm = A.batchDot(text1Embed, text2Embed, axes = List(2, 2)) // Translation Matrix.
    val KM = new ArrayBuffer[Variable[T]]()
    for (i <- 0 until kernelNum) {
      var mu = 1.0 / (kernelNum - 1) + (2.0 * i) / (kernelNum - 1) - 1.0
      val _sigma = if (mu > 1.0) { // Exact match.
        mu = 1.0
        exactSigma
      } else sigma
      val mmExp = A.exp[T]((mm - mu) * (mm - mu) / _sigma / _sigma * (-0.5))
      val mmDocSum = A.sum(mmExp, axis = 2)
      val mmLog = A.log(mmDocSum + 1.0)
      // Remark: Keep the reduced dimension for the last sum and squeeze after stack.
      // Otherwise, when batch=1, the output will become a Scalar not compatible stack.
      val mmSum = A.sum(mmLog, axis = 1, keepDims = true)
      KM.append(mmSum)
    }
    val Phi = Squeeze(2).inputs(A.stack(KM.toList).node)
    val output = if (targetMode == "ranking") Dense(1, init = "uniform").inputs(Phi)
    else Dense(1, init = "uniform", activation = "sigmoid").inputs(Phi)
    Model(input, output)
  }
}

object KNRM {
  /**
   * The factory method to create a KNRM instance using embeddingFile and wordIndex.
   *
   * @param embeddingFile The path to the word embedding file.
   *                      Currently only the following GloVe files are supported:
   *                      "glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt",
   *                      "glove.6B.300d.txt", "glove.42B.300d.txt", "glove.840B.300d.txt".
   *                      You can download from: https://nlp.stanford.edu/projects/glove/.
   * @param wordIndex Map of word (String) and its corresponding index (integer).
   *                  The index is supposed to start from 1 with 0 reserved for unknown words.
   *                  During the prediction, if you have words that are not in the wordIndex
   *                  for the training, you can map them to index 0.
   *                  Default is null. In this case, all the words in the embeddingFile will
   *                  be taken into account and you can call
   *                  WordEmbedding.getWordIndex(embeddingFile) to retrieve the map.
   */
  def apply[@specialized(Float, Double) T: ClassTag](
      text1Length: Int,
      text2Length: Int,
      embeddingFile: String,
      wordIndex: Map[String, Int] = null,
      trainEmbed: Boolean = true,
      kernelNum: Int = 21,
      sigma: Double = 0.1,
      exactSigma: Double = 0.001,
      targetMode: String = "ranking")(implicit ev: TensorNumeric[T]): KNRM[T] = {
    val (vocabSize, embedSize, embedWeights) = WordEmbedding.prepareEmbedding[T](
      embeddingFile, wordIndex, randomizeUnknown = true, normalize = true)
    new KNRM[T](text1Length, text2Length, vocabSize, embedSize, embedWeights,
      trainEmbed, kernelNum, sigma, exactSigma, targetMode).build()
  }

  def apply[@specialized(Float, Double) T: ClassTag](
      text1Length: Int,
      text2Length: Int,
      vocabSize: Int,
      embedSize: Int,
      embedWeights: Tensor[T],
      trainEmbed: Boolean,
      kernelNum: Int,
      sigma: Double,
      exactSigma: Double,
      targetMode: String)(implicit ev: TensorNumeric[T]): KNRM[T] = {
    new KNRM[T](text1Length, text2Length, vocabSize, embedSize, embedWeights,
      trainEmbed, kernelNum, sigma, exactSigma, targetMode).build()
  }

  /**
   * This factory method is mainly for Python use.
   * Pass in a model to build the KNRM model.
   * Note that if you use this factory method, arguments should match the model definition
   * to eliminate ambiguity.
   */
  private[zoo] def apply[@specialized(Float, Double) T: ClassTag](
      text1Length: Int,
      text2Length: Int,
      vocabSize: Int,
      embedSize: Int,
      embedWeights: Tensor[T],
      trainEmbed: Boolean,
      kernelNum: Int,
      sigma: Double,
      exactSigma: Double,
      targetMode: String,
      model: AbstractModule[Activity, Activity, T])
    (implicit ev: TensorNumeric[T]): KNRM[T] = {
    new KNRM[T](text1Length, text2Length, vocabSize, embedSize, embedWeights,
      trainEmbed, kernelNum, sigma, exactSigma, targetMode).addModel(model)
  }

  /**
   * Load an existing KNRM model (with weights).
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
      weightPath: String = null)(implicit ev: TensorNumeric[T]): KNRM[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[KNRM[T]]
  }
}

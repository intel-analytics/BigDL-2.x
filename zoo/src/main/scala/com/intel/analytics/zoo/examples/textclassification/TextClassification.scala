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

package com.intel.analytics.zoo.examples.textclassification

import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.text.TextSet
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.log4j.{Level => Level4j, Logger => Logger4j}
import scopt.OptionParser


case class TextClassificationParams(
   dataPath: String = "./", embeddingPath: String = "./",
   classNum: Int = 20, tokenLength: Int = 200,
   sequenceLength: Int = 500, encoder: String = "cnn",
   encoderOutputDim: Int = 256, maxWordsNum: Int = 5000,
   trainingSplit: Double = 0.8, batchSize: Int = 128,
   nbEpoch: Int = 20, learningRate: Double = 0.01,
   partitionNum: Int = 4, model: Option[String] = None,
   outputPath: Option[String] = None)


object TextClassification {
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level4j.INFO)

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[TextClassificationParams]("TextClassification Example") {
      opt[String]("dataPath")
        .required()
        .text("The directory containing the training data")
        .action((x, c) => c.copy(dataPath = x))
      opt[String]("embeddingPath")
        .required()
        .text("The directory for GloVe embeddings")
        .action((x, c) => c.copy(embeddingPath = x))
      opt[Int]("classNum")
        .text("The number of classes to do classification")
        .action((x, c) => c.copy(classNum = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]("tokenLength")
        .text("The size of each word vector, 50 or 100 or 200 or 300 for GloVe")
        .action((x, c) => c.copy(tokenLength = x))
      opt[Int]("sequenceLength")
        .text("The length of each sequence")
        .action((x, c) => c.copy(sequenceLength = x))
      opt[Int]("maxWordsNum")
        .text("The maximum number of words to be taken into consideration")
        .action((x, c) => c.copy(maxWordsNum = x))
      opt[String]("encoder")
        .text("The encoder for the input sequence, cnn or lstm or gru")
        .action((x, c) => c.copy(encoder = x))
      opt[Int]("encoderOutputDim")
        .text("The output dimension of the encoder")
        .action((x, c) => c.copy(encoderOutputDim = x))
      opt[Double]("trainingSplit")
        .text("The split portion of the data for training")
        .action((x, c) => c.copy(trainingSplit = x))
      opt[Int]('b', "batchSize")
        .text("The number of samples per gradient update")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]('e', "nbEpoch")
        .text("The number of epochs to train the model")
        .action((x, c) => c.copy(nbEpoch = x))
      opt[Double]('l', "learningRate")
        .text("The learning rate for the TextClassifier model")
        .action((x, c) => c.copy(learningRate = x))
      opt[String]('m', "model")
        .text("Model snapshot location if any")
        .action((x, c) => c.copy(model = Some(x)))
      opt[String]('o', "outputPath")
        .text("The directory to save the model and word dictionary")
        .action((x, c) => c.copy(outputPath = Some(x)))
    }

    parser.parse(args, TextClassificationParams()).map { param =>
      val sc = NNContext.initNNContext("Text Classification Example")

      val textSet = TextSet.read(param.dataPath)
        .toDistributed(sc, param.partitionNum)
      println("Processing text dataset...")
      val transformed = textSet.tokenize().normalize()
        .word2idx(removeTopN = 10, maxWordsNum = param.maxWordsNum)
        .shapeSequence(param.sequenceLength).generateSample()
      val Array(trainTextSet, valTextSet) = transformed.randomSplit(
        Array(param.trainingSplit, 1 - param.trainingSplit))

      val model = if (param.model.isDefined) {
        TextClassifier.loadModel(param.model.get)
      }
      else {
        val tokenLength = param.tokenLength
        require(tokenLength == 50 || tokenLength == 100 || tokenLength == 200 || tokenLength == 300,
        s"tokenLength for GloVe can only be 50, 100, 200, 300, but got $tokenLength")
        val wordIndex = transformed.getWordIndex
        val gloveFile = param.embeddingPath + "/glove.6B." + tokenLength.toString + "d.txt"
        TextClassifier(param.classNum, gloveFile, wordIndex, param.sequenceLength,
          param.encoder, param.encoderOutputDim)
      }

      model.compile(
        optimizer = new Adagrad(learningRate = param.learningRate,
          learningRateDecay = 0.001),
        loss = SparseCategoricalCrossEntropy[Float](),
        metrics = List(new Accuracy()))
      model.fit(trainTextSet, batchSize = param.batchSize,
        nbEpoch = param.nbEpoch, validationData = valTextSet)

      val predictSet = model.predict(valTextSet, batchPerThread = param.partitionNum)
      println("Probability distributions of the first five texts in the validation set:")
      predictSet.toDistributed().rdd.take(5).foreach(feature => {
        println("Prediction for " + feature.getURI + ": ")
        println(feature.getPredict.toTensor)
      })
      if (param.outputPath.isDefined) {
        val outputPath = param.outputPath.get
        model.saveModel(outputPath + "/text_classifier.model")
        transformed.saveWordIndex(outputPath + "/word_index.txt")
        println("Trained model and word dictionary saved")
      }
      sc.stop()
    }
  }
}

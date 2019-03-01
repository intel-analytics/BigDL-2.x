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

package com.intel.analytics.zoo.examples.qaranker

import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.layers.TimeDistributed
import com.intel.analytics.zoo.pipeline.api.keras.objectives.RankHinge
import com.intel.analytics.zoo.models.textmatching.KNRM
import com.intel.analytics.zoo.feature.common.Relations
import com.intel.analytics.zoo.feature.pmem.MemoryType
import com.intel.analytics.zoo.feature.text.TextSet
import scopt.OptionParser


case class QARankerParams(
    dataPath: String = "./", embeddingFile: String = "./",
    questionLength: Int = 10, answerLength: Int = 40,
    partitionNum: Int = 4, batchSize: Int = 200,
    nbEpoch: Int = 30, learningRate: Double = 0.001,
    model: Option[String] = None, memoryType: String = "DRAM")


object QARanker {
  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[QARankerParams]("QARanker Example") {
      opt[String]("dataPath")
        .required()
        .text("The directory containing the corpus and relations")
        .action((x, c) => c.copy(dataPath = x))
      opt[String]("embeddingFile")
        .required()
        .text("The file path to GloVe embeddings")
        .action((x, c) => c.copy(embeddingFile = x))
      opt[Int]("questionLength")
        .text("The sequence length of each question")
        .action((x, c) => c.copy(questionLength = x))
      opt[Int]("answerLength")
        .text("The sequence length of each answer")
        .action((x, c) => c.copy(answerLength = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]('b', "batchSize")
        .text("The number of samples per gradient update")
        .action((x, c) => c.copy(batchSize = x))
      opt[Int]('e', "nbEpoch")
        .text("The number of epochs to train the model")
        .action((x, c) => c.copy(nbEpoch = x))
      opt[Double]('l', "learningRate")
        .text("The learning rate for the model")
        .action((x, c) => c.copy(learningRate = x))
      opt[String]('m', "model")
        .text("KNRM model snapshot location if any")
        .action((x, c) => c.copy(model = Some(x)))
      opt[String]("memoryType")
        .text("memory type")
        .action((x, c) => c.copy(memoryType = x))
    }

    parser.parse(args, QARankerParams()).map { param =>
      val sc = NNContext.initNNContext("QARanker Example")

      val qSet = TextSet.readCSV(param.dataPath + "/question_corpus.csv", sc, param.partitionNum)
        .tokenize().normalize().word2idx(minFreq = 2).shapeSequence(param.questionLength)
      val aSet = TextSet.readCSV(param.dataPath + "/answer_corpus.csv", sc, param.partitionNum)
        .tokenize().normalize().word2idx(minFreq = 2, existingMap = qSet.getWordIndex)
        .shapeSequence(param.answerLength)

      val trainRelations = Relations.read(param.dataPath + "/relation_train.csv",
        sc, param.partitionNum)
      val trainSet = TextSet.fromRelationPairs(trainRelations, qSet, aSet,
        MemoryType.fromString(param.memoryType))
      val validateRelations = Relations.read(param.dataPath + "/relation_valid.csv",
        sc, param.partitionNum)
      val validateSet = TextSet.fromRelationLists(validateRelations, qSet, aSet)

      val knrm = if (param.model.isDefined) {
        KNRM.loadModel(param.model.get)
      } else {
        val wordIndex = aSet.getWordIndex
        KNRM(param.questionLength, param.answerLength, param.embeddingFile, wordIndex)
      }
      val model = Sequential().add(TimeDistributed(
        knrm, inputShape = Shape(2, param.questionLength + param.answerLength)))
      model.compile(optimizer = new SGD(learningRate = param.learningRate),
        loss = RankHinge[Float]())
      for (i <- 1 to param.nbEpoch) {
        model.fit(trainSet, batchSize = param.batchSize, nbEpoch = 1)
        knrm.evaluateNDCG(validateSet, 3)
        knrm.evaluateNDCG(validateSet, 5)
        knrm.evaluateMAP(validateSet)
      }
      sc.stop()
    }
  }
}

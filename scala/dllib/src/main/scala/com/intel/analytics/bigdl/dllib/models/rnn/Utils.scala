/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.models.rnn
import scopt.OptionParser
import java.io._

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.util.Random
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.io.Source
import scala.reflect.ClassTag

object Utils {

  case class TrainParams(
                          dataFolder: String = "./",
                          saveFolder: String = "./",
                          modelSnapshot: Option[String] = None,
                          stateSnapshot: Option[String] = None,
                          checkpoint: Option[String] = None,
                          batchSize: Int = 128,
                          learningRate: Double = 0.1,
                          momentum: Double = 0.0,
                          weightDecay: Double = 0.0,
                          dampening: Double = 0.0,
                          hiddenSize: Int = 40,
                          vocabSize: Int = 4000,
                          bptt: Int = 4,
                          nEpochs: Int = 30,
                          sentFile: Option[String] = None,
                          tokenFile: Option[String] = None,
                          overWriteCheckpoint: Boolean = false)

  val trainParser = new OptionParser[TrainParams]("BigDL SimpleRNN Train Example") {
    opt[String]('f', "dataFolder")
      .text("where you put the text data")
      .action((x, c) => c.copy(dataFolder = x))
      .required()

    opt[String]('s', "saveFolder")
      .text("where you save the processed text data")
      .action((x, c) => c.copy(saveFolder = x))
      .required()

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))

    opt[String]("checkpoint")
      .text("where to cache the model and state")
      .action((x, c) => c.copy(checkpoint = Some(x)))

    opt[Int]('b', "batchSize")
      .text("batchSize of rnn")
      .action((x, c) => c.copy(batchSize = x))
      .required()

    opt[Double]('r', "learningRate")
      .text("learning rate")
      .action((x, c) => c.copy(learningRate = x))

    opt[Double]('m', "momentum")
      .text("momentum")
      .action((x, c) => c.copy(momentum = x))

    opt[Double]("weightDecay")
      .text("weight decay")
      .action((x, c) => c.copy(weightDecay = x))

    opt[Double]("dampening")
      .text("dampening")
      .action((x, c) => c.copy(dampening = x))

    opt[Int]('h', "hidden")
      .text("hidden size")
      .action((x, c) => c.copy(hiddenSize = x))

    opt[Int]("vocab")
      .text("dictionary length | vocabulary size")
      .action((x, c) => c.copy(vocabSize = x))

    opt[Int]("bptt")
      .text("back propagation through time size")
      .action((x, c) => c.copy(bptt = x))

    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))

    opt[String]("sent")
      .text("sentence dictionary to split document into sentences")
      .action((x, c) => c.copy(sentFile = Some(x)))

    opt[String]("token")
      .text("token dictionary to split sentence into tokens")
      .action((x, c) => c.copy(tokenFile = Some(x)))

    opt[Unit]("overWrite")
      .text("overwrite checkpoint files")
      .action( (_, c) => c.copy(overWriteCheckpoint = true) )
  }

  case class TestParams(
     folder: String = "./",
     modelSnapshot: Option[String] = None,
     numOfWords: Option[Int] = None,
     batchSize: Int = 2
  )

  val testParser = new OptionParser[TestParams]("BigDL rnn Test Example") {
    opt[String]('f', "folder")
      .text("where you put the dictionary data")
      .action((x, c) => c.copy(folder = x))
      .required()

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
      .required()

    opt[Int]("words")
      .text("number of words to write")
      .action((x, c) => c.copy(numOfWords = Some(x)))
      .required()

    opt[Int]('b', "batchSize")
      .text("batchSize of rnn")
      .action((x, c) => c.copy(batchSize = x))
      .required()
  }

  private[bigdl] def readSentence(directory: String)
  : Array[Array[String]] = {

    import scala.io.Source
    require(new File(directory + "/test.txt").exists(),
      s"test file ${directory + "/test.txt"} not exists!")
    val lines = Source.fromFile(directory + "/test.txt")
      .getLines().map(_.split("\\W+")).toArray
    lines
  }
}

object SequencePreprocess {
  def apply(
    fileName: String,
    sc: SparkContext,
    sentBin: Option[String],
    tokenBin: Option[String])
  : RDD[Array[String]] = {

    val sentenceSplitter = SentenceSplitter(sentBin)
    val sentenceTokenizer = SentenceTokenizer(tokenBin)
    val lines = load(fileName)
    val tokens = DataSet.array(lines, sc)
      .transform(sentenceSplitter).toDistributed().data(false)
      .flatMap(x => x).mapPartitions(x => SentenceBiPadding().apply(x))
      .mapPartitions(x => sentenceTokenizer.apply(x))
    tokens
  }

  private[bigdl] def load(fileName: String): Array[String] = {
    import scala.io.Source
    require(new File(fileName).exists(),
      s"data file ${fileName} not exists!")
    val lines = Source.fromFile(fileName).getLines().toArray
    lines
  }
}

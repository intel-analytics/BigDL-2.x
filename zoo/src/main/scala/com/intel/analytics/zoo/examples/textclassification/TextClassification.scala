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

import java.io.File
import java.util

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.example.utils.SimpleTokenizer._
import com.intel.analytics.bigdl.example.utils.{SimpleTokenizer, WordMeta}
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

import com.intel.analytics.bigdl.utils.LoggerFilter
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import org.apache.log4j.{Level => Level4j, Logger => Logger4j}
import org.apache.spark.SparkConf

import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}
import scopt.OptionParser

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source

case class TextClassificationParams(baseDir: String = "./",
                                    tokenLength: Int = 200,
                                    sequenceLength: Int = 500,
                                    encoder: String = "cnn",
                                    encoderOutputDim: Int = 256,
                                    maxWordsNum: Int = 5000,
                                    trainingSplit: Double = 0.8,
                                    batchSize: Int = 128,
                                    nbEpoch: Int = 20,
                                    learningRate: Double = 0.01,
                                    partitionNum: Int = 4,
                                    model: Option[String] = None)

object TextClassification {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level4j.INFO)

  var classNum: Int = -1

  // Load text, label pairs from file
  def loadRawData(dir: String): ArrayBuffer[(String, Float)] = {
    val texts = ArrayBuffer[String]()
    val labels = ArrayBuffer[Float]()
    // Category is a string name and label is it's one-based index
    val categoryToLabel = new util.HashMap[String, Int]()
    val categoryPathList = new File(dir).listFiles().filter(_.isDirectory).toList.sorted

    categoryPathList.foreach { categoryPath =>
      val label_id = categoryToLabel.size() + 1
      categoryToLabel.put(categoryPath.getName(), label_id)
      val textFiles = categoryPath.listFiles()
        .filter(_.isFile).filter(_.getName.forall(Character.isDigit(_))).sorted
      textFiles.foreach { file =>
        val source = Source.fromFile(file, "ISO-8859-1")
        val text = try source.getLines().toList.mkString("\n") finally source.close()
        texts.append(text)
        labels.append(label_id)
      }
    }
    this.classNum = labels.toSet.size
    log.info(s"Found ${texts.length} texts.")
    log.info(s"Found $classNum classes")
    texts.zip(labels)
  }

  // Turn texts into tokens
  def analyzeTexts(dataRdd: RDD[(String, Float)], maxWordsNum: Int, gloveDir: String)
  : (Map[String, WordMeta], Map[Float, Array[Float]]) = {
    // Remove the top 10 words roughly. You might want to fine tune this.
    val frequencies = dataRdd.flatMap{case (text: String, label: Float) =>
      SimpleTokenizer.toTokens(text)
    }.map(word => (word, 1)).reduceByKey(_ + _)
      .sortBy(- _._2).collect().slice(10, maxWordsNum)

    val indexes = Range(1, frequencies.length)
    val word2Meta = frequencies.zip(indexes).map{item =>
      (item._1._1, WordMeta(item._1._2, item._2))}.toMap
    (word2Meta, buildWord2Vec(word2Meta, gloveDir))
  }

  // Turn tokens into vectors
  def buildWord2Vec(word2Meta: Map[String, WordMeta], gloveDir: String):
      Map[Float, Array[Float]] = {
    log.info("Indexing word vectors.")
    val preWord2Vec = MMap[Float, Array[Float]]()
    val filename = s"$gloveDir/glove.6B.200d.txt"
    for (line <- Source.fromFile(filename, "ISO-8859-1").getLines) {
      val values = line.split(" ")
      val word = values(0)
      if (word2Meta.contains(word)) {
        val coefs = values.slice(1, values.length).map(_.toFloat)
        preWord2Vec.put(word2Meta(word).index.toFloat, coefs)
      }
    }
    log.info(s"Found ${preWord2Vec.size} word vectors.")
    preWord2Vec.toMap
  }

  def main(args: Array[String]): Unit = {
    val parser = new OptionParser[TextClassificationParams]("TextClassification Example") {
      opt[String]("baseDir")
        .required()
        .text("The base directory containing the training and word2Vec data")
        .action((x, c) => c.copy(baseDir = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]("tokenLength")
        .text("The size of each word vector")
        .action((x, c) => c.copy(tokenLength = x))
      opt[Int]("sequenceLength")
        .text("The length of a sequence")
        .action((x, c) => c.copy(sequenceLength = x))
      opt[Int]("maxWordsNum")
        .text("The maximum number of words")
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
      opt[Int]("nbEpoch")
        .text("The number of iterations to train the model")
        .action((x, c) => c.copy(nbEpoch = x))
      opt[Double]('l', "learningRate")
        .text("The learning rate for the TextClassifier model")
        .action((x, c) => c.copy(learningRate = x))
      opt[String]("model")
        .text("Model snapshot location if any")
        .action((x, c) => c.copy(model = Some(x)))
    }

    parser.parse(args, TextClassificationParams()).map { param =>
      val conf = new SparkConf()
        .setAppName("Text Classification Example")
        .set("spark.task.maxFailures", "1")
      val sc = NNContext.initNNContext(conf)

      val sequenceLength = param.sequenceLength
      val tokenLength = param.tokenLength
      val trainingSplit = param.trainingSplit
      val textDataDir = s"${param.baseDir}/20news-18828/"
      require(new File(textDataDir).exists(), "Text data directory is not found in baseDir, " +
        "you can run $ANALYTICS_ZOO_HOME/bin/data/news20/get_news20.sh to " +
        "download 20 Newsgroup dataset")
      val gloveDir = s"${param.baseDir}/glove.6B/"
      require(new File(gloveDir).exists(),
        "Glove word embeddings directory is not found in baseDir, " +
        "you can run $ANALYTICS_ZOO_HOME/bin/data/glove/get_glove.sh to download")

      // For large dataset, you might want to get such RDD[(String, Float)] from HDFS
      val dataRdd = sc.parallelize(loadRawData(textDataDir), param.partitionNum)
      val (word2Meta, word2Vec) = analyzeTexts(dataRdd, param.maxWordsNum, gloveDir)
      val word2MetaBC = sc.broadcast(word2Meta)
      val word2VecBC = sc.broadcast(word2Vec)
      val vectorizedRdd = dataRdd
        .map {case (text, label) => (toTokens(text, word2MetaBC.value), label)}
        .map {case (tokens, label) => (shaping(tokens, sequenceLength), label)}
        .map {case (tokens, label) => (vectorization(
          tokens, tokenLength, word2VecBC.value), label)}
      val sampleRDD = vectorizedRdd.map {case (input: Array[Array[Float]], label: Float) =>
        Sample(
          featureTensor = Tensor(input.flatten, Array(sequenceLength, tokenLength)),
          label = label)
      }

      val Array(trainingRDD, valRDD) = sampleRDD.randomSplit(
        Array(trainingSplit, 1 - trainingSplit))

      val model = if (param.model.isDefined) {
        TextClassifier.loadModel(param.model.get)
      }
      else {
        TextClassifier(classNum, tokenLength, sequenceLength,
          param.encoder, param.encoderOutputDim)
      }

      val optimizer = Optimizer(
        model = model,
        sampleRDD = trainingRDD,
        criterion = ClassNLLCriterion[Float](logProbAsInput = false),
        batchSize = param.batchSize
      )

      optimizer
        .setOptimMethod(new Adagrad(learningRate = param.learningRate,
          learningRateDecay = 0.001))
        .setValidation(Trigger.everyEpoch, valRDD, Array(new Top1Accuracy), param.batchSize)
        .setEndWhen(Trigger.maxEpoch(param.nbEpoch))
        .optimize()

      // Predict for probability distributions
      val results = model.predict(valRDD)
      // Predict for labels
      val resultClasses = model.predictClass(valRDD)

      sc.stop()
    }
  }
}

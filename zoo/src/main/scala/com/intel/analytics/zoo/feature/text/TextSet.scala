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

package com.intel.analytics.zoo.feature.text

import com.intel.analytics.zoo.feature.common.Preprocessing
import org.apache.spark.ml.Transformer
import org.apache.spark.rdd.RDD

/**
 * TextSet wraps a set of TextFeature.
 */
abstract class TextSet {

  // The very first TextSet instance that haven't been but should be transformed by
  // a PipelinedSparkNLPTransformer specified by stages.
  val preTextSet: TextSet = null

  // The stages that can construct a PipelinedSparkNLPTransformer, which can transform preTextSet
  // to the target TextSet in the current status.
  // With preTextSet and stages, we can get the current TextSet.
  // The idea is similar to _prev_jrdd and func in pyspark PipelinedRDD implementation.
  val stages: Array[Transformer] = null

  def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    this.transform(transformer)
  }

  def isLocal(): Boolean

  def isDistributed(): Boolean

  def toLocal(): LocalTextSet = this.asInstanceOf[LocalTextSet]

  def toDistributed(): DistributedTextSet = this.asInstanceOf[DistributedTextSet]

  def tokenize(): TextSet = {
    transform(Tokenizer())
  }

  def normalize(): TextSet = {
    transform(Normalizer())
  }

  def token2idx(removeTopN: Int = 0, maxWordsNum: Int = 5000): TextSet = {
    val map = generateWordIndexMap(removeTopN, maxWordsNum)
    transform(WordIndexer(map)).setWordIndex(map)
  }

  def shapeSequence(
    len: Int,
    trunc: String = "pre",
    key: String = "indexedTokens"): TextSet = {
    transform(SequenceShaper(len, trunc, key))
  }

  def genSample(): TextSet = {
    transform(TextSetToSample())
  }

  def generateWordIndexMap(
     removeTopN: Int = 0, maxWordsNum: Int = 5000): Map[String, Int]

  private var wordIndex: Map[String, Int] = _

  // TODO: throw exception for null
  def getWordIndex: Map[String, Int] = wordIndex

  def setWordIndex(map: Map[String, Int]): this.type = {
    wordIndex = map
    this
  }

  // Return preTextSet and stages for the new TextSet after applying a SparkNLPTransformer.
  // In this case, the corresponding RDD/Array should be null as PipelinedSparkNLPTransformer
  // has been applied yet.
  protected def processSparkNLPTransformer(
    transformer: SparkNLPTransformer): (TextSet, Array[Transformer]) = {
    if (preTextSet == null) {
      (this, Array(transformer.labor))
    }
    else {
      require(stages != null)
      (preTextSet, stages ++ Array(transformer.labor))
    }
  }
}


object TextSet {

  def array(data: Array[TextFeature]): LocalTextSet = {
    new LocalTextSet(data)
  }

  def rdd(data: RDD[TextFeature]): DistributedTextSet = {
    new DistributedTextSet(data)
  }

  // Given an array of words, each with its frequency, sorted descending by frequency,
  // return a Map of word and its index.
  // Index starts from 1.
  def wordIndexFromFrequencies(frequencies: Array[(String, Int)]): Map[String, Int] = {
    val indexes = Range(1, frequencies.length + 1)
    frequencies.zip(indexes).map{item =>
      (item._1._1, item._2)}.toMap
  }
}


class LocalTextSet(var array: Array[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    val (preT, pipelineStages, curArr) = transformer match {
      case sparkNLP: SparkNLPTransformer =>
        val (t, s) = processSparkNLPTransformer(sparkNLP)
        (t, s, null)
      case nonSparkNLP =>
        if (array != null) (null, null, nonSparkNLP.apply(array.toIterator).toArray)
        else (null, null, nonSparkNLP.apply(sparkNLPTransformArray.toIterator).toArray)
    }
    new LocalTextSet(curArr) {
      override val preTextSet: TextSet = preT
      override val stages: Array[Transformer] = pipelineStages
    }.setWordIndex(getWordIndex)
  }

  override def isLocal(): Boolean = true

  override def isDistributed(): Boolean = false

  override def token2idx(removeTopN: Int = 0, maxWordsNum: Int = 5000): TextSet = {
    if (array == null) {
      array = sparkNLPTransformArray
    }
    super.token2idx(removeTopN, maxWordsNum)
  }

  override def generateWordIndexMap(
    removeTopN: Int = 0, maxWordsNum: Int = 5000): Map[String, Int] = {
    val frequencies = array.flatMap(feature => feature.apply[Array[String]]("tokens"))
      .groupBy(identity).mapValues(_.length)
      .toArray.sortBy(- _._2).slice(removeTopN, maxWordsNum + removeTopN)
    val res = TextSet.wordIndexFromFrequencies(frequencies)
    setWordIndex(res)
    res
  }

  private def sparkNLPTransformArray: Array[TextFeature] = {
    preTextSet.transform(PipelinedSparkNLPTransformer(stages))
      .asInstanceOf[LocalTextSet].array
  }
}


class DistributedTextSet(var rdd: RDD[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    val (preT, pipelineStages, curRDD) = transformer match {
      case sparkNLP: SparkNLPTransformer =>
        val (t, s) = processSparkNLPTransformer(sparkNLP)
        (t, s, null)
      case nonSparkNLP =>
        // Two cases when rdd != null:
        // either prevTextSet == null or transformed by calling indexize
        if (rdd != null) (null, null, nonSparkNLP(rdd))
        else (null, null, nonSparkNLP(sparkNLPTransformRDD))
    }
    new DistributedTextSet(curRDD) {
      override val preTextSet: TextSet = preT
      override val stages: Array[Transformer] = pipelineStages
    }.setWordIndex(getWordIndex)
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

  override def token2idx(removeTopN: Int = 0, maxWordsNum: Int = 5000): TextSet = {
    if (rdd == null) {
      rdd = sparkNLPTransformRDD
    }
    super.token2idx(removeTopN, maxWordsNum)
  }

  override def generateWordIndexMap(
    removeTopN: Int = 0, maxWordsNum: Int = 5000): Map[String, Int] = {
    val frequencies = rdd.flatMap(text => text.apply[Array[String]]("tokens"))
      .map(word => (word, 1)).reduceByKey(_ + _)
      .sortBy(- _._2).collect().slice(removeTopN, maxWordsNum + removeTopN)
    val res = TextSet.wordIndexFromFrequencies(frequencies)
    setWordIndex(res)
    res
  }

  private def sparkNLPTransformRDD: RDD[TextFeature] = {
    preTextSet.transform(PipelinedSparkNLPTransformer(stages))
      .asInstanceOf[DistributedTextSet].rdd
  }
}

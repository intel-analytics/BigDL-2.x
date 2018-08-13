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

abstract class TextSet {

  // The very first TextSet object that haven't been transformed by SparkNLPTransformer
  val preTextSet: TextSet = null

  // The stages that can construct a pipelined SparkNLP transformer, which can transform
  // preTextSet to the current target TextSet.
  // With preTextSet and stages, we can get the current TextSet.
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

  def indexize(removeTopN: Int = 0, maxWordsNum: Int = 5000): TextSet = {
    val wordIndex = getWordIndexByFrequencies(removeTopN, maxWordsNum)
    transform(WordIndexer(wordIndex)).setWordIndex(wordIndex)
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

  def getWordIndexByFrequencies(
     removeTopN: Int = 0, maxWordsNum: Int = 5000): Map[String, Int]

  protected var wordIndex: Map[String, Int] = _

  def getWordIndex: Map[String, Int] = wordIndex

  def setWordIndex(map: Map[String, Int]): this.type = {
    wordIndex = map
    this
  }

}


object TextSet {

  def array(data: Array[TextFeature]): LocalTextSet = {
    new LocalTextSet(data)
  }

  def rdd(data: RDD[TextFeature]): DistributedTextSet = {
    new DistributedTextSet(data)
  }
}


class LocalTextSet(var array: Array[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    array = transformer.apply(array.toIterator).toArray
    this
  }

  override def isLocal(): Boolean = true

  override def isDistributed(): Boolean = false

  override def getWordIndexByFrequencies(
    removeTopN: Int = 0, maxWordsNum: Int = 5000): Map[String, Int] = {
    val frequencies = array.flatMap(feature => feature.apply[Array[String]]("tokens"))
      .map(word => (word, 1))
    // TODO: finish this part
    null
  }
}


class DistributedTextSet(var rdd: RDD[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    val (prev, pipeline, curRDD) = if (transformer.isInstanceOf[SparkNLPTransformer]) {
      if (preTextSet == null) {
        (this, Array(transformer.asInstanceOf[SparkNLPTransformer].transformer), null)
      }
      else {
        (preTextSet,
          stages ++ Array(transformer.asInstanceOf[SparkNLPTransformer].transformer),
        null)
      }
    }
    else {
      if (rdd != null) { // either prevTextSet == null or transformed by calling indexize
        (null, null, transformer(rdd))
      }
      else {
        (null, null, transformer(sparkNLPTransformRDD))
      }
    }
    new DistributedTextSet(curRDD) {
      override val preTextSet: TextSet = prev
      override val stages: Array[Transformer] = pipeline
    }.setWordIndex(getWordIndex)
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

  override def indexize(removeTopN: Int = 0, maxWordsNum: Int = 5000): TextSet = {
    if (rdd == null) {
      rdd = sparkNLPTransformRDD
    }
    super.indexize(removeTopN, maxWordsNum)
  }

  override def getWordIndexByFrequencies(
    removeTopN: Int = 0, maxWordsNum: Int = 5000): Map[String, Int] = {
    val frequencies = rdd.flatMap(text => text.apply[Array[String]]("tokens"))
      .map(word => (word, 1)).reduceByKey(_ + _)
      .sortBy(- _._2).collect().slice(removeTopN, maxWordsNum + removeTopN)
    val indexes = Range(1, frequencies.length + 1)
    val res = frequencies.zip(indexes).map{item =>
      (item._1._1, item._2)}.toMap
    setWordIndex(res)
    res
  }

  private def sparkNLPTransformRDD: RDD[TextFeature] = {
    preTextSet.transform(PipelinedSparkNLPTransformer(stages))
      .asInstanceOf[DistributedTextSet].rdd
  }

}

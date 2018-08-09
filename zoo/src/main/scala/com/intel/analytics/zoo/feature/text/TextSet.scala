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
import org.apache.spark.rdd.RDD

abstract class TextSet {

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
    this.transform(Tokenizer())
  }

  def indexize(removeTopN: Int = 10, maxWordsNum: Int = 5000): TextSet = {
    setWordIndex(getWordIndexByFrequencies(removeTopN, maxWordsNum))
    this.transform(WordIndexer(wordIndex))
  }

  def shapeSequence(
    len: Int,
    trunc: String = "pre",
    key: String = "indexedTokens"): TextSet = {
    this.transform(SequenceShaper(len, trunc, key))
  }

  def genSample(): TextSet = {
    this.transform(TextSetToSample())
  }

  def getWordIndexByFrequencies(
     removeTopN: Int = 10, maxWordsNum: Int = 5000): Map[String, Int]

  private var wordIndex: Map[String, Int] = null

  def getWordIndex(): Map[String, Int] = wordIndex

  protected def setWordIndex(map: Map[String, Int]): Unit = {
    wordIndex = map
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
    removeTopN: Int = 10, maxWordsNum: Int = 5000): Map[String, Int] = {
    val frequencies = array.flatMap(feature => feature.apply[Array[String]]("tokens"))
      .map(word => (word, 1))
    // TODO: finish this part
    null
  }
}


class DistributedTextSet(var rdd: RDD[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

  override def getWordIndexByFrequencies(
    removeTopN: Int = 10, maxWordsNum: Int = 5000): Map[String, Int] = {
    val frequencies = rdd.flatMap(text => text.apply[Array[String]]("tokens"))
      .map(word => (word, 1)).reduceByKey(_ + _)
      .sortBy(- _._2).collect().slice(removeTopN, maxWordsNum)
    val indexes = Range(1, frequencies.length + 1)
    val res = frequencies.zip(indexes).map{item =>
      (item._1._1, item._2)}.toMap
    setWordIndex(res)
    res
  }

}

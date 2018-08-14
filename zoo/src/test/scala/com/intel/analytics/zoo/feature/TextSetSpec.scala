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

package com.intel.analytics.zoo.feature

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.text._
import org.apache.spark.SparkConf
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.immutable.HashSet

class TextSetSpec extends FlatSpec with Matchers {
  val text1 = TextFeature("Hello my friend, please annotate my text", label = 0)
  val text2 = TextFeature("Listen to my heart Heart. Show me love, baby.", label = 1)

  "TextFeature properties" should "work properly" in {
    require(text1.getText == "Hello my friend, please annotate my text")
    require(text1.hasLabel)
    require(text1.getLabel == 0)
    require(text1.keys() == HashSet("label", "text"))
  }

  "DistributedTextSet Transformation" should "work properly" in {
    val conf = new SparkConf().setAppName("Test TextSet").setMaster("local[*]")
    val sc = NNContext.initNNContext(conf)
    val distributed = TextSet.rdd(sc.parallelize(Seq(text1, text2)))
    require(distributed.isDistributed)
    require(distributed.preTextSet == null)
    require(distributed.stages == null)
    val t1 = distributed.tokenize()
    require(t1.preTextSet == distributed)
    require(t1.stages.length == 1 &&
      t1.stages.head.isInstanceOf[com.johnsnowlabs.nlp.annotators.Tokenizer])
    val t2 = t1.normalize()
    require(t2.preTextSet == distributed)
    require(t2.stages.length == 2)
    require(t2.stages(0).isInstanceOf[com.johnsnowlabs.nlp.annotators.Tokenizer])
    require(t2.stages(1).isInstanceOf[com.johnsnowlabs.nlp.annotators.NormalizerModel])
    val t3 = t2.word2idx(maxWordsNum = 10)
    require(t3.preTextSet == null)
    require(t3.stages == null)
    val t4 = t3.shapeSequence(len = 5).genSample()
    require(t4.isDistributed)
    require(t4.preTextSet == null)
    require(t4.stages == null)
    val wordIndex1 = t3.getWordIndex
    val wordIndex2 = t4.getWordIndex
    require(wordIndex1 == wordIndex2 && wordIndex2.toArray.length == 10)
    require(wordIndex1("my") == 1)
    val arr = t4.toDistributed.rdd.collect()
    require(arr.length == 2)
    require(arr(0).keys() == HashSet("label", "text", "tokens", "indexedTokens", "sample"))
    require(arr(0).apply[Array[Int]]("indexedTokens").length == 5)
  }

  "LocalTextSet Transformation" should "work properly" in {
    val local = TextSet.array(Array(text1, text2))
    require(local.isLocal)
    require(local.preTextSet == null)
    require(local.stages == null)
    val transformed =
      local.tokenize().normalize().word2idx().shapeSequence(len = 6).genSample()
    require(transformed.isLocal)
    require(transformed.preTextSet == null)
    require(transformed.stages == null)
    val wordIndex = transformed.getWordIndex
    require(wordIndex.toArray.length == 13)
    require(wordIndex.keySet.contains("heart"))
    require(!wordIndex.keySet.contains("Heart"))
    require(wordIndex("my") == 1)
    val arr = transformed.toLocal.array
    require(arr.length == 2)
    require(arr(0).keys() == HashSet("label", "text", "tokens", "indexedTokens", "sample"))
    require(arr(0).apply[Array[Int]]("indexedTokens").length == 6)
  }
}

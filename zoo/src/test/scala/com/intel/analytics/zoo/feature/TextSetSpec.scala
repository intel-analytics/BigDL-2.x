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
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.immutable.HashSet

class TextSetSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val text1 = new TextFeature("Hello my friend, please annotate my text", Some(0))
  val text2 = new TextFeature("Listen to my heart Heart. Show me love, baby.", Some(1))
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test TextFeature").setMaster("local[*]")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  "DistributedTextSet Transformation" should "work properly" in {
    val distributed = TextSet.rdd(sc.parallelize(Seq(text1, text2)))
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
    val t3 = t2.indexize(maxWordsNum = 10)
    require(t3.preTextSet == null)
    require(t3.stages == null)
    val t4 = t3.shapeSequence(len = 5).genSample()
    require(t4.preTextSet == null)
    require(t4.stages == null)
    val wordIndex1 = t3.getWordIndex
    val wordIndex2 = t4.getWordIndex
    require(wordIndex1 == wordIndex2 && wordIndex2.toArray.length == 10)
    require(wordIndex1("my") == 1)
    val arr = t4.asInstanceOf[DistributedTextSet].rdd.collect()
    require(arr.length == 2)
    require(arr(0).keys() == HashSet("label", "text", "tokens", "indexedTokens", "sample"))
    require(arr(0).apply[Array[Int]]("indexedTokens").length == 5)
  }

  "LocalTextSet Transformation" should "work properly" in {
    val local = TextSet.array(Array(text1, text2))
    require(local.preTextSet == null)
    require(local.stages == null)
    val transformed =
      local.tokenize().normalize().indexize().shapeSequence(len = 6).genSample()
    require(transformed.preTextSet == null)
    require(transformed.stages == null)
    val wordIndex = transformed.getWordIndex
    require(wordIndex.toArray.length == 13)
    require(wordIndex.keySet.contains("heart"))
    require(!wordIndex.keySet.contains("Heart"))
    require(wordIndex("my") == 1)
    val arr = transformed.asInstanceOf[LocalTextSet].array
    require(arr.length == 2)
    require(arr(0).keys() == HashSet("label", "text", "tokens", "indexedTokens", "sample"))
    require(arr(0).apply[Array[Int]]("indexedTokens").length == 6)
  }
}

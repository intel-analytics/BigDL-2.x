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

class TextSetSpec extends FlatSpec with Matchers with BeforeAndAfter {
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
    val text1 = new TextFeature("Hello my friend, please annotate my text", Some(0))
    val text2 = new TextFeature("Listen to my heart Heart. Show me love, baby.", Some(1))
    val textSet = TextSet.rdd(sc.parallelize(Seq(text1, text2)))
    val transformedTextSet =
      textSet.tokenize().normalize().indexize().shapeSequence(len = 6).genSample()
    val transformedTextFeatures = transformedTextSet.asInstanceOf[DistributedTextSet].rdd.collect()
    val wordIndex = transformedTextSet.getWordIndex
    wordIndex.toArray.length should be (13)
    wordIndex.keySet.contains("heart") should be (true)
    wordIndex.keySet.contains("Heart") should be (false)
    wordIndex("my") should be (1)
    val transformed1 = transformedTextFeatures(0)
    transformed1.keys().toArray.length should be (5)
    transformed1.keys().contains("sample") should be (true)
    transformed1.apply[Array[Int]]("indexedTokens").length should be (6)
  }
}

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

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Convolution1D, Dense, Embedding, Flatten}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.immutable.HashSet

class TextSetSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val text1 = "Hello my friend, please annotate my text"
  val text2 = "hello world, this is some sentence for my test"
  val text3 = "dummy text for test"
  val feature1 = TextFeature(text1, label = 0)
  val feature2 = TextFeature(text2, label = 1)
  val feature3 = TextFeature(text3)
  val path: String = getClass.getClassLoader.getResource("news20").getPath
  var sc : SparkContext = _

  before {
    val conf = new SparkConf().setAppName("Test TextFeature and TextSet").setMaster("local[4]")
    sc = NNContext.initNNContext(conf)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  def buildModel(): Sequential[Float] = {
    val model = Sequential()
    model.add(Embedding(300, 20, inputLength = 30))
    model.add(Convolution1D(8, 4))
    model.add(Flatten())
    model.add(Dense(3, activation = "softmax"))
    model
  }

  "TextFeature with label" should "work properly" in {
    require(feature1.getText == "Hello my friend, please annotate my text")
    require(feature1.hasLabel)
    require(feature1.getLabel == 0)
    require(feature1.keys() == HashSet("label", "text"))
  }

  "TextFeature without label" should "work properly" in {
    require(!feature3.hasLabel)
    require(feature3.getLabel == -1)
    require(feature3.keys() == HashSet("text"))
  }

  "DistributedTextSet Transformation" should "work properly" in {
    val distributed = TextSet.rdd(sc.parallelize(Seq(feature1, feature2)))
    require(distributed.isDistributed)
    require(distributed.preTextSet == null)
    require(distributed.stages == null)

    // After tokenization
    val t1 = distributed.tokenize()
    require(t1.preTextSet == distributed)
    require(t1.stages.length == 1 && t1.stages.head.isInstanceOf[Tokenizer])
    require(t1.toDistributed.rdd == null)

    // After normalization
    val t2 = t1.normalize()
    require(t2.preTextSet == distributed)
    require(t2.stages.length == 2)
    require(t2.stages.head.isInstanceOf[Tokenizer])
    require(t2.stages.last.isInstanceOf[Normalizer])
    require(t2.toDistributed.rdd == null)

    // After wordToIndex
    val t3 = t2.word2idx(maxWordsNum = 10)
    require(t3.preTextSet == null)
    require(t3.stages == null)

    // After shaping and generating sample
    val t4 = t3.shapeSequence(len = 5).genSample[Float]()
    require(t4.isDistributed)
    require(t4.preTextSet == null)
    require(t4.stages == null)

    val wordIndex1 = t3.getWordIndex
    val wordIndex2 = t4.getWordIndex
    require(wordIndex1 == wordIndex2 && wordIndex2.toArray.length == 10)
    require(wordIndex1("my") == 1)

    val features = t4.toDistributed.rdd.collect()
    require(features.length == 2)
    require(features(0).keys() == HashSet("label", "text", "tokens", "indexedTokens", "sample"))
    require(features(0).apply[Array[Int]]("indexedTokens").length == 5)
  }

  "LocalTextSet Transformation" should "work properly" in {
    val local = TextSet.array(Array(feature1, feature2))
    require(local.isLocal)
    require(local.preTextSet == null)
    require(local.stages == null)

    val transformed =
      local.tokenize().normalize().word2idx().shapeSequence(len = 6).genSample[Float]()
    require(transformed.isLocal)
    require(transformed.preTextSet == null)
    require(transformed.stages == null)

    val wordIndex = transformed.getWordIndex
    require(wordIndex.toArray.length == 13)
    require(wordIndex.keySet.contains("hello"))
    require(!wordIndex.keySet.contains("Hello"))
    require(wordIndex("my") == 1)

    val features = transformed.toLocal.array
    require(features.length == 2)
    require(features(0).keys() == HashSet("label", "text", "tokens", "indexedTokens", "sample"))
    require(features(0).apply[Array[Int]]("indexedTokens").length == 6)
  }

  "TextSet read with sc, fit, predict and evaluate" should "work properly" in {
    val textSet = TextSet.read(path, sc)
    require(textSet.isDistributed)
    require(textSet.toDistributed.rdd.count() == 5)
    require(textSet.toDistributed.rdd.collect().head.keys() == HashSet("label", "text"))
    val transformed = textSet.tokenize().normalize()
      .word2idx(removeTopN = 5, maxWordsNum = 299).shapeSequence(len = 30).genSample()
    val model = buildModel()
    model.compile("sgd", "sparse_categorical_crossentropy", List("accuracy"))
    model.fit(transformed, batchSize = 4, nbEpoch = 2, validationData = transformed)
    require(! transformed.toDistributed.rdd.first().contains("predict"))
    val predictSet = model.predict(transformed, batchPerThread = 1)
    require(predictSet.toDistributed.rdd.first().contains("predict"))
    val accuracy = model.evaluate(transformed, batchSize = 4)
  }

  "TextSet read without sc" should "work properly" in {
    val textSet = TextSet.read(path)
    require(textSet.isLocal)
    require(textSet.toLocal.array.length == 5)
    require(textSet.toLocal.array.head.keys() == HashSet("label", "text"))
    val transformed = textSet.tokenize().normalize()
      .word2idx(removeTopN = 5, maxWordsNum = 299).shapeSequence(len = 30).genSample()
    val model = buildModel()
    model.compile("sgd", "sparse_categorical_crossentropy", List("accuracy"))
    model.fit(transformed, batchSize = 4, nbEpoch = 2, validationData = transformed)
    require(! transformed.toLocal.array.head.contains("predict"))
    val predictSet = model.predict(transformed, batchPerThread = 1)
    require(predictSet.toLocal.array.head.contains("predict"))
    val accuracy = model.evaluate(transformed, batchSize = 4)
  }
}
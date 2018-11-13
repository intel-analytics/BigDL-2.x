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

import com.intel.analytics.bigdl.optim.{Adagrad, SGD}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.metrics.Accuracy
import com.intel.analytics.zoo.pipeline.api.keras.objectives.SparseCategoricalCrossEntropy
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.HashSet

class TextSetSpec extends ZooSpecHelper {
  val text1 = "Hello my friend, please annotate my text"
  val text2 = "hello world, this is some sentence for my test"
  val path: String = getClass.getClassLoader.getResource("news20").getPath
  var sc : SparkContext = _
  val gloveDir: String = getClass.getClassLoader.getResource("glove.6B").getPath
  val embeddingFile: String = gloveDir + "/glove.6B.50d.txt"

  override def doBefore(): Unit = {
    val conf = new SparkConf().setAppName("Test TextSet").setMaster("local[1]")
    sc = NNContext.initNNContext(conf)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  private def genFeatures(): Array[TextFeature] = {
    val feature1 = TextFeature(text1, label = 0)
    val feature2 = TextFeature(text2, label = 1)
    Array(feature1, feature2)
  }

  "DistributedTextSet Transformation" should "work properly" in {
    val distributed = TextSet.rdd(sc.parallelize(genFeatures()))
    require(distributed.isDistributed)
    val shaped = distributed -> Tokenizer() -> Normalizer() -> SequenceShaper(len = 5)
    val transformed = shaped.word2idx().generateSample()
    require(transformed.isDistributed)

    val wordIndex = transformed.getWordIndex
    require(wordIndex.toArray.length == 9)
    require(wordIndex.keySet == HashSet("friend", "please", "annotate", "my", "text",
      "some", "sentence", "for", "test"))
    require(wordIndex("my") == 1)

    val features = transformed.toDistributed().rdd.collect()
    require(features.length == 2)
    require(features(0).keys() == HashSet("label", "text", "tokens", "indexedTokens", "sample"))
    require(features(0)[Array[Float]]("indexedTokens").length == 5)
  }

  "LocalTextSet Transformation" should "work properly" in {
    val local = TextSet.array(genFeatures())
    require(local.isLocal)
    val transformed = local.tokenize().normalize().shapeSequence(len = 10)
      .word2idx(removeTopN = 1).generateSample()
    require(transformed.isLocal)

    val wordIndex = transformed.getWordIndex
    require(wordIndex.toArray.length == 12)
    require(wordIndex.keySet.contains("hello"))
    require(!wordIndex.keySet.contains("Hello"))
    require(!wordIndex.keySet.contains("##"))

    val features = transformed.toLocal().array
    require(features.length == 2)
    require(features(0).keys() == HashSet("label", "text", "tokens", "indexedTokens", "sample"))
    require(features(0)[Array[Float]]("indexedTokens").length == 10)
  }

  "TextSet read with sc, fit, predict and evaluate" should "work properly" in {
    val textSet = TextSet.read(path, sc)
    require(textSet.isDistributed)
    require(textSet.toDistributed().rdd.count() == 5)
    require(textSet.toDistributed().rdd.collect().head.keys() == HashSet("label", "text"))
    val transformed = textSet.tokenize().normalize()
      .shapeSequence(len = 30).word2idx().generateSample()
    val model = TextClassifier(3, embeddingFile, transformed.getWordIndex, 30)
    model.compile(new SGD[Float](), SparseCategoricalCrossEntropy[Float](), List(new Accuracy()))
    model.fit(transformed, batchSize = 4, nbEpoch = 2, validationData = transformed)
    require(! transformed.toDistributed().rdd.first().contains("predict"))

    val predictSet = model.predict(transformed, batchPerThread = 2).toDistributed()
    val textFeatures = predictSet.rdd.collect()
    textFeatures.foreach(feature => {
      require(feature.contains("predict"))
      val input = feature.getSample.feature.reshape(Array(1, 30))
      val output = model.setEvaluateStatus().forward(input).toTensor[Float].split(1)(0)
      feature.getPredict[Float] should be (output)
    })
    val accuracy = model.evaluate(transformed, batchSize = 4)

    // Test for loaded model predict on TextSet
    val saveFile = createTmpFile()
    model.saveModel(saveFile.getAbsolutePath, overWrite = true)
    val loadedModel = TextClassifier.loadModel[Float](saveFile.getAbsolutePath)
    val predictResults = model.predict(transformed, batchPerThread = 2)
      .toDistributed().rdd.collect()
  }

  "TextSet read without sc, fit, predict and evaluate" should "work properly" in {
    val textSet = TextSet.read(path)
    require(textSet.isLocal)
    require(textSet.toLocal().array.length == 5)
    require(textSet.toLocal().array.head.keys() == HashSet("label", "text"))
    val tokenized = textSet -> Tokenizer() -> Normalizer() -> SequenceShaper(len = 30)
    val wordIndex = tokenized.generateWordIndexMap()
    val transformed = tokenized -> WordIndexer(wordIndex) -> TextFeatureToSample()
    require(transformed.getWordIndex == wordIndex)
    val model = TextClassifier(10, embeddingFile, wordIndex, 30)
    model.compile(new Adagrad[Float](), SparseCategoricalCrossEntropy[Float](),
      List(new Accuracy()))
    model.fit(transformed, batchSize = 4, nbEpoch = 2, validationData = transformed)
    require(! transformed.toLocal().array.head.contains("predict"))

    val predictSet = model.predict(transformed, batchPerThread = 2).toLocal()
    val textFeatures = predictSet.array
    textFeatures.foreach(feature => {
      require(feature.contains("predict"))
      val input = feature.getSample.feature.reshape(Array(1, 30))
      val output = model.setEvaluateStatus().forward(input).toTensor[Float].split(1)(0)
      feature.getPredict[Float] should be(output)
    })
    val accuracy = model.evaluate(transformed, batchSize = 4)

    val saveFile = createTmpFile()
    model.saveModel(saveFile.getAbsolutePath, overWrite = true)
    val loadedModel = TextClassifier.loadModel[Float](saveFile.getAbsolutePath)
    val predictResults = model.predict(transformed, batchPerThread = 2).toLocal().array
  }
}

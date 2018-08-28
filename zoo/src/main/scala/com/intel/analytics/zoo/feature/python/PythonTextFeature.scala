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

package com.intel.analytics.zoo.feature.python

import java.util.{List => JList, Map => JMap}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.python.api.{JTensor, PythonBigDL, Sample}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dataset.{Sample => JSample}
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.text._
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

object PythonTextFeature {

  def ofFloat(): PythonTextFeature[Float] = new PythonTextFeature[Float]()

  def ofDouble(): PythonTextFeature[Double] = new PythonTextFeature[Double]()
}

class PythonTextFeature[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

  def createTextFeature(text: String): TextFeature = {
    TextFeature(text)
  }

  def createTextFeature(text: String, label: Int): TextFeature = {
    TextFeature(text, label)
  }

  def textFeatureGetText(feature: TextFeature): String = {
    feature.getText
  }

  def textFeatureGetLabel(feature: TextFeature): Int = {
    feature.getLabel
  }

  def textFeatureHasLabel(feature: TextFeature): Boolean = {
    feature.hasLabel
  }

  def textFeatureGetKeys(feature: TextFeature): JList[String] = {
    feature.keys().toList.asJava
  }

  def createLocalTextSet(texts: JList[String], labels: JList[Int]): LocalTextSet = {
    require(texts != null, "texts of a TextSet can't be null")
    val features = if (labels != null) {
      require(texts.size() == labels.size(), "texts and labels of a TextSet " +
        "should have the same size")
      texts.asScala.toArray[String].zip(labels.asScala.toArray[Int]).map{feature =>
        createTextFeature(feature._1, feature._2)
      }
    }
    else {
      texts.asScala.toArray.map(createTextFeature)
    }
    TextSet.array(features)
  }

  def createDistributedTextSet(texts: JavaRDD[String], labels: JavaRDD[Int]):
  DistributedTextSet = {
    require(texts != null, "texts of a TextSet can't be null")
    val features = if (labels != null) {
      texts.rdd.zip(labels.rdd).map{feature =>
        createTextFeature(feature._1, feature._2)
      }
    }
    else {
      texts.rdd.map(createTextFeature)
    }
    TextSet.rdd(features)
  }

  def readTextSet(path: String, sc: JavaSparkContext, minPartitions: Int): TextSet = {
    if (sc == null) {
      TextSet.read(path, null, minPartitions)
    }
    else {
      TextSet.read(path, sc.sc, minPartitions)
    }
  }

  def textSetGetWordIndex(textSet: TextSet): JMap[String, Int] = {
    val res = textSet.getWordIndex
    if (res == null) {
      null
    }
    else {
      res.asJava
    }
  }

  def textSetIsDistributed(textSet: TextSet): Boolean = {
    textSet.isDistributed
  }

  def textSetIsLocal(textSet: TextSet): Boolean = {
    textSet.isLocal
  }

  def transformTextSet(
      transformer: Preprocessing[TextFeature, TextFeature],
      imageSet: TextSet): TextSet = {
    imageSet.transform(transformer)
  }

  def localTextSetGetTexts(textSet: LocalTextSet): JList[String] = {
    textSet.array.map(_.getText).toList.asJava
  }

  def localTextSetGetLabels(textSet: LocalTextSet): JList[Int] = {
    textSet.array.map(_.getLabel).toList.asJava
  }

  def localTextSetGetPredicts(textSet: LocalTextSet): JList[JList[JTensor]] = {
    textSet.array.map{feature =>
      if (feature.contains(TextFeature.predict)) {
        activityToJTensors(feature[Activity](TextFeature.predict))
      }
      else {
        null
      }
    }.toList.asJava
  }

  def localTextSetGetSamples(textSet: LocalTextSet): JList[Sample] = {
    textSet.array.map{feature =>
      if (feature.contains(TextFeature.predict)) {
        toPySample(feature[JSample[T]](TextFeature.sample))
      }
      else {
        null
      }
    }.toList.asJava
  }

  def distributedTextSetGetTexts(textSet: DistributedTextSet): JavaRDD[String] = {
    textSet.rdd.map(_.getText).toJavaRDD()
  }

  def distributedTextSetGetLabels(textSet: DistributedTextSet): JavaRDD[Int] = {
    textSet.rdd.map(_.getLabel).toJavaRDD()
  }

  def distributedTextSetGetPredicts(textSet: DistributedTextSet): JavaRDD[JList[JTensor]] = {
    textSet.rdd.map{feature =>
      if (feature.contains(TextFeature.predict)) {
        activityToJTensors(feature[Activity](TextFeature.predict))
      }
      else {
        null
      }
    }.toJavaRDD()
  }

  def distributedTextSetGetSamples(textSet: DistributedTextSet): JavaRDD[Sample] = {
    textSet.rdd.map{feature =>
      if (feature.contains(TextFeature.sample)) {
        toPySample(feature[JSample[T]](TextFeature.sample))
      }
      else {
        null
      }
    }.toJavaRDD()
  }

  def createTokenizer(outKey: String): Tokenizer = {
    Tokenizer(outKey)
  }

  def createNormalizer(outKey: String): Normalizer = {
    Normalizer(outKey)
  }

  def createWordIndexer(map: JMap[String, Int]): WordIndexer = {
    WordIndexer(map.asScala.toMap)
  }

  def createSequenceShaper(
      len: Int,
      mode: String,
      inputKey: String,
      padElement: Int): SequenceShaper = {
    SequenceShaper(len, mode, inputKey, padElement)
  }

  def createSequenceShaper(
      len: Int,
      mode: String,
      inputKey: String,
      padElement: String): SequenceShaper = {
    SequenceShaper(len, mode, inputKey, padElement)
  }

  def createTextFeatureToSample(): TextFeatureToSample[T] = {
    TextFeatureToSample[T]()
  }

  def textSetRandomSplit(
      textSet: TextSet,
      weights: JList[Double]): JList[TextSet] = {
    textSet.randomSplit(weights.asScala.toArray).toList.asJava
  }

  def textSetTokenize(textSet: TextSet): TextSet = {
    textSet.tokenize()
  }

  def textSetNormalize(textSet: TextSet): TextSet = {
    textSet.normalize()
  }

  def textSetWord2idx(
      textSet: TextSet,
      removeTopN: Int,
      maxWordsNum: Int): TextSet = {
    textSet.word2idx(removeTopN, maxWordsNum)
  }

  def textSetShapeSequence(
      textSet: TextSet,
      len: Int,
      mode: String,
      inputKey: String,
      padElement: Int): TextSet = {
    textSet.shapeSequence(len, mode, inputKey, padElement)
  }

  def textSetShapeSequence(
      textSet: TextSet,
      len: Int,
      mode: String,
      key: String,
      padElement: String): TextSet = {
    textSet.shapeSequence(len, mode, key, padElement)
  }

  def textSetGenSample(textSet: TextSet): TextSet = {
    textSet.genSample[T]()
  }

}

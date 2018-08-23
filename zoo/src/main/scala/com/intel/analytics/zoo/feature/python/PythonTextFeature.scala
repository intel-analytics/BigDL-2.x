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

import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
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

  def distributedTextSetGetTexts(textSet: DistributedTextSet): JavaRDD[String] = {
    textSet.rdd.map(_.getText).toJavaRDD()
  }

  def distributedTextSetGetLabels(textSet: DistributedTextSet): JavaRDD[Int] = {
    textSet.rdd.map(_.getLabel).toJavaRDD()
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

  def createSequenceShaper(len: Int, truncMode: String, inputKey: String): SequenceShaper = {
    SequenceShaper(len, truncMode, inputKey)
  }

  def createTextFeatureToSample(): TextFeatureToSample[T] = {
    TextFeatureToSample[T]()
  }

}

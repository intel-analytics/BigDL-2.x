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

import java.io.File
import java.util

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.feature.common.Preprocessing
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.ml.Transformer
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.reflect.ClassTag

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

  def isLocal: Boolean

  def isDistributed: Boolean

  def toLocal: LocalTextSet = this.asInstanceOf[LocalTextSet]

  def toDistributed: DistributedTextSet = this.asInstanceOf[DistributedTextSet]

  /**
   * Convert TextSet to DataSet of TextFeature.
   */
  def toDataSet: DataSet[TextFeature]

  def tokenize(): TextSet = {
    transform(Tokenizer())
  }

  def normalize(): TextSet = {
    transform(Normalizer())
  }

  def word2idx(removeTopN: Int = 0, maxWordsNum: Int = 5000): TextSet = {
    val map = if (wordIndex != null) {
      TextSet.logger.warn("wordIndex already exists. Using the existing wordIndex")
      wordIndex
    } else {
      generateWordIndexMap(removeTopN, maxWordsNum)
    }
    transform(WordIndexer(map)).setWordIndex(map)
  }

  def shapeSequence(
    len: Int,
    trunc: String = "pre",
    key: String = "indexedTokens"): TextSet = {
    transform(SequenceShaper(len, trunc, key))
  }

  def genSample[T: ClassTag]()(implicit ev: TensorNumeric[T]): TextSet = {
    transform(TextFeatureToSample[T]())
  }

  def generateWordIndexMap(
     removeTopN: Int = 0, maxWordsNum: Int = 5000): Map[String, Int]

  private var wordIndex: Map[String, Int] = _

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

  val logger: Logger = Logger.getLogger(getClass)

  /**
   * Create a LocalTextSet from TextFeature array.
   */
  def array(data: Array[TextFeature]): LocalTextSet = {
    new LocalTextSet(data)
  }

  /**
   * Create a DistributedTextSet from TextFeature RDD.
   */
  def rdd(data: RDD[TextFeature]): DistributedTextSet = {
    new DistributedTextSet(data)
  }

  /**
   * Read text files as TextSet.
   * If sc is defined, read texts as DistributedTextSet from local file system or HDFS.
   * If sc is null, read texts as LocalTextSet from local file system.
   *
   * @param path String. Folder path to texts. The folder structure is expected to be the following:
   *             path
   *                ├── dir1 - text1, text2, ...
   *                ├── dir2 - text1, text2, ...
   *                └── dir3 - text1, text2, ...
   *             Under the target path, there ought to be N subdirectories. Each subdirectory
   *             represents a category and contains all available texts of such category.
   *             Each category will be a given a label according to its position in the
   *             ascending order sort among all subdirectories.
   *             All texts will be given a label according to the subdirectory it is located.
   *             Labels start from 0.
   * @param sc An instance of SparkContext if any.
   * @param minPartitions A suggestion value of the minimal splitting number for input data.
   *                      Integer. Default is 1.
   * @return A TextSet.
   */
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1): TextSet = {
    val textSet = if (sc != null) {
      val fs = FileSystem.get(new Configuration())
      val categories = fs.listStatus(new Path(path)).map(_.getPath.getName).sorted
      logger.info(s"Found ${categories.length} classes.")
      // Labels of categories start from 0.
      val indices = categories.indices
      val categoryToLabel = categories.zip(indices).toMap
      val textRDD = sc.wholeTextFiles(path + "/*", minPartitions).map{case (p, text) =>
        val parts = p.split("/")
        val category = parts(parts.length - 2)
        TextFeature(text, label = categoryToLabel(category))
      }
      TextSet.rdd(textRDD)
    }
    else {
      val texts = ArrayBuffer[String]()
      val labels = ArrayBuffer[Int]()
      val categoryToLabel = new util.HashMap[String, Int]()
      val categoryPathList = new File(path).listFiles().filter(_.isDirectory).toList.sorted
      categoryPathList.foreach { categoryPath =>
        val label = categoryToLabel.size()
        categoryToLabel.put(categoryPath.getName, label)
        val textFiles = categoryPath.listFiles()
          .filter(_.isFile).filter(_.getName.forall(Character.isDigit(_))).sorted
        textFiles.foreach { file =>
          val source = Source.fromFile(file, "ISO-8859-1")
          val text = try source.getLines().toList.mkString("\n") finally source.close()
          texts.append(text)
          labels.append(label)
        }
      }
      val textArr = texts.zip(labels).map{case (text, label) =>
          TextFeature(text, label)
      }.toArray
      TextSet.array(textArr)
    }
    textSet
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

  override def isLocal: Boolean = true

  override def isDistributed: Boolean = false

  override def toDataSet: DataSet[TextFeature] = {
    DataSet.array(array)
  }

  override def word2idx(removeTopN: Int = 0, maxWordsNum: Int = 5000): TextSet = {
    if (array == null) {
      array = sparkNLPTransformArray
    }
    super.word2idx(removeTopN, maxWordsNum)
  }

  override def generateWordIndexMap(
    removeTopN: Int = 0, maxWordsNum: Int = 5000): Map[String, Int] = {
    val frequencies = array.flatMap(feature => feature[Array[String]](TextFeature.tokens))
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

  override def isLocal: Boolean = false

  override def isDistributed: Boolean = true

  override def toDataSet: DataSet[TextFeature] = {
    DataSet.rdd[TextFeature](rdd)
  }

  override def word2idx(removeTopN: Int = 0, maxWordsNum: Int = 5000): TextSet = {
    if (rdd == null) {
      rdd = sparkNLPTransformRDD
    }
    super.word2idx(removeTopN, maxWordsNum)
  }

  override def generateWordIndexMap(
    removeTopN: Int = 0, maxWordsNum: Int = 5000): Map[String, Int] = {
    val frequencies = rdd.flatMap(text => text[Array[String]](TextFeature.tokens))
      .map(word => (word, 1)).reduceByKey(_ + _)
      .sortBy(- _._2).collect().slice(removeTopN, maxWordsNum + removeTopN)
    rdd.cache()
    val res = TextSet.wordIndexFromFrequencies(frequencies)
    setWordIndex(res)
    res
  }

  private def sparkNLPTransformRDD: RDD[TextFeature] = {
    preTextSet.transform(PipelinedSparkNLPTransformer(stages))
      .asInstanceOf[DistributedTextSet].rdd
  }
}

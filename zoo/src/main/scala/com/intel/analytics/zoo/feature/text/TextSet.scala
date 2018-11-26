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
import com.intel.analytics.bigdl.dataset.{DataSet, Sample}
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.feature.text.TruncMode.TruncMode
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.util.StringUtils
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import java.io.PrintWriter

import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.spark.sql.SQLContext

/**
 * TextSet wraps a set of TextFeature.
 */
abstract class TextSet {
  import TextSet.logger

  /**
   * Transform from one TextSet to another.
   */
  def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    this.transform(transformer)
  }

  /**
   * Whether it is a LocalTextSet.
   */
  def isLocal: Boolean

  /**
   * Whether it is a DistributedTextSet.
   */
  def isDistributed: Boolean

  /**
   * Convert to a LocalTextSet.
   */
  def toLocal(): LocalTextSet

  /**
   * Convert to a DistributedTextSet.
   *
   * Need to specify SparkContext to convert a LocalTextSet to a DistributedTextSet.
   * In this case, you may also want to specify partitionNum, the default of which is 4.
   */
  def toDistributed(sc: SparkContext = null, partitionNum: Int = 4): DistributedTextSet

  /**
   * Convert TextSet to DataSet of Sample.
   */
  def toDataSet: DataSet[Sample[Float]]

  /**
   * Randomly split into array of TextSet with provided weights.
   * Only available for DistributedTextSet for now.
   *
   * @param weights Array of Double indicating the split portions.
   */
  def randomSplit(weights: Array[Double]): Array[TextSet]

  /**
   * Do tokenization on original text.
   * See Tokenizer for more details.
   */
  def tokenize(): TextSet = {
    transform(Tokenizer())
  }

  /**
   * Do normalization on tokens.
   * Need to tokenize first.
   * See Normalizer for more details.
   */
  def normalize(): TextSet = {
    transform(Normalizer())
  }

  /**
   * Map word tokens to indices.
   * Result index will start from 1 and corresponds to the occurrence frequency of each word
   * sorted in descending order.
   * Need to tokenize first.
   * See WordIndexer for more details.
   * After word2idx, you can get the generated wordIndex map by calling 'getWordIndex'.
   *
   * @param removeTopN Non-negative Integer. Remove the topN words with highest frequencies in
   *                   the case where those are treated as stopwords.
   *                   Default is 0, namely remove nothing.
   * @param maxWordsNum Integer. The maximum number of words to be taken into consideration.
   *                    Default is -1, namely all words will be considered.
   */
  def word2idx(
    removeTopN: Int = 0,
    maxWordsNum: Int = -1,
    minFreq: Int = 1,
    existingMap: Map[String, Int] = null): TextSet = {
    if (wordIndex != null) {
      logger.warn("wordIndex already exists. Using the existing wordIndex")
    } else {
      generateWordIndexMap(removeTopN, maxWordsNum, minFreq, existingMap)
    }
    transform(WordIndexer(wordIndex))
  }

  /**
   * Shape the sequence of indices to a fixed length.
   * Need to word2idx first.
   * See SequenceShaper for more details.
   */
  def shapeSequence(
      len: Int,
      truncMode: TruncMode = TruncMode.pre,
      padElement: Int = 0): TextSet = {
    transform(SequenceShaper(len, truncMode, padElement))
  }

  /**
   * Generate BigDL Sample.
   * Need to word2idx first.
   * See TextFeatureToSample for more details.
   */
  def generateSample(): TextSet = {
    transform(TextFeatureToSample())
  }

  /**
   * Generate wordIndex map based on sorted word frequencies in descending order.
   * Return the result map, which will also be stored in 'wordIndex'.
   * Make sure you call this after tokenize. Otherwise you will get an exception.
   * See word2idx for more details.
   */
  def generateWordIndexMap(
      removeTopN: Int = 0,
      maxWordsNum: Int = 5000,
      minFreq: Int = 1,
      existingMap: Map[String, Int] = null): Map[String, Int]

  private var wordIndex: Map[String, Int] = _

  /**
   * Get the word index map of this TextSet.
   * If the TextSet hasn't been transformed from word to index, null will be returned.
   */
  def getWordIndex: Map[String, Int] = wordIndex

  def setWordIndex(map: Map[String, Int]): this.type = {
    wordIndex = map
    this
  }

  def saveWordIndex(path: String): Unit = {
    if (wordIndex == null) {
      logger.warn("wordIndex is null, please transform from word to index first")
    }
    else {
      val pw = new PrintWriter(new File(path))
      for (item <- wordIndex) {
        pw.print(item._1)
        pw.print(" ")
        pw.println(item._2)
      }
      pw.close()
    }
  }
}


object TextSet {

  val logger: Logger = Logger.getLogger(getClass)

  /**
   * Create a LocalTextSet from array of TextFeature.
   */
  def array(data: Array[TextFeature]): LocalTextSet = {
    new LocalTextSet(data)
  }

  /**
   * Create a DistributedTextSet from RDD of TextFeature.
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
   *             Under the target path, there ought to be N subdirectories (dir1 to dirN). Each
   *             subdirectory represents a category and contains all texts that belong to such
   *             category. Each category will be a given a label according to its position in the
   *             ascending order sorted among all subdirectories.
   *             All texts will be given a label according to the subdirectory where it is located.
   *             Labels start from 0.
   * @param sc An instance of SparkContext.
   *           If specified, texts will be read as a DistributedTextSet.
   *           Default is null and in this case texts will be read as a LocalTextSet.
   * @param minPartitions Integer. A suggestion value of the minimal partition number for input
   *                      texts. Only need to specify this when sc is not null. Default is 1.
   * @return TextSet.
   */
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1): TextSet = {
    val textSet = if (sc != null) {
      // URI needs for the FileSystem to accept HDFS
      val uri = StringUtils.stringToURI(Array(path))(0)
      val fs = FileSystem.get(uri, new Configuration())
      val categories = fs.listStatus(new Path(path)).map(_.getPath.getName).sorted
      logger.info(s"Found ${categories.length} classes.")
      // Labels of categories start from 0.
      val indices = categories.indices
      val categoryToLabel = categories.zip(indices).toMap
      val textRDD = sc.wholeTextFiles(path + "/*", minPartitions).map{case (p, text) =>
        val parts = p.split("/")
        val category = parts(parts.length - 2)
        TextFeature(text, label = categoryToLabel(category), uri = p)
      }
      TextSet.rdd(textRDD)
    }
    else {
      val features = ArrayBuffer[TextFeature]()
      val categoryToLabel = new util.HashMap[String, Int]()
      val categoryPath = new File(path)
      require(categoryPath.exists(), s"$path doesn't exist. Please check your input path")
      val categoryPathList = categoryPath.listFiles().filter(_.isDirectory).toList.sorted
      categoryPathList.foreach { categoryPath =>
        val label = categoryToLabel.size()
        categoryToLabel.put(categoryPath.getName, label)
        val textFiles = categoryPath.listFiles()
          .filter(_.isFile).filter(_.getName.forall(Character.isDigit(_))).sorted
        textFiles.foreach { file =>
          val source = Source.fromFile(file, "ISO-8859-1")
          val text = try source.getLines().toList.mkString("\n") finally source.close()
          features.append(TextFeature(text, label, file.getAbsolutePath))
        }
      }
      logger.info(s"Found ${categoryToLabel.size()} classes")
      TextSet.array(features.toArray)
    }
    textSet
  }

  // Without header
  // ID, content
  def readCSV(path: String, sc: SparkContext = null, minPartitions: Int = 1): TextSet = {
    if (sc != null) {
      val textRDD = sc.textFile(path, minPartitions).map(line => {
        val subs = line.split(",", 2) // "," may exist in content.
        TextFeature(subs(1), uri = subs(0))
      })
      TextSet.rdd(textRDD)
    }
    else {
      val src = Source.fromFile(path)
      val textArray = src.getLines().toArray.map(line => {
        val subs = line.split(",", 2)
        TextFeature(subs(1), uri = subs(0))
      })
      TextSet.array(textArray)
    }
  }

  def readParquet(path: String, sqlContext: SQLContext): DistributedTextSet = {
    val textRDD = sqlContext.read.parquet(path).rdd.map(row => {
      val uri = row.getAs[String](TextFeature.uri)
      val text = row.getAs[String](TextFeature.text)
      TextFeature(text, uri = uri)
    })
    TextSet.rdd(textRDD)
  }

  // Generate RelationPairs: (text1ID, text2PosID, text2NegID)
  // and transform each RelationPair to a TextFeature.
  def fromRelationPairs(
      relations: RDD[Relation],
      text1Corpus: TextSet,
      text2Corpus: TextSet): TextSet = {
    val pairsRDD = Relations.generateRelationPairs(relations)
    require(text1Corpus.isDistributed, "text1Corpus must be a DistributedTextSet")
    require(text2Corpus.isDistributed, "text2Corpus must be a DistributedTextSet")
    val joinedText1 = text1Corpus.toDistributed().rdd.keyBy(_.uri())
      .join(pairsRDD.keyBy(_.text1ID)).map(_._2)
    val joinedText2Pos = text2Corpus.toDistributed().rdd.keyBy(_.uri())
      .join(joinedText1.keyBy(_._2.text2PosID)).map(x => (x._2._2._1, x._2._1, x._2._2._2))
    val joinedText2Neg = text2Corpus.toDistributed().rdd.keyBy(_.uri())
      .join(joinedText2Pos.keyBy(_._3.text2NegID))
      .map(x => (x._2._2._1, x._2._2._2, x._2._1))
    val res = joinedText2Neg.map(x => {
      val textFeature = TextFeature(null, x._1.uri() + x._2.uri() + x._3.uri())
      val text1 = x._1.getIndices
      val text2Pos = x._2.getIndices
      val text2Neg = x._3.getIndices
      require(text1 != null,
        "text1Corpus haven't been transformed from word to index yet, please word2idx first")
      require(text2Pos != null && text2Neg != null,
        "text2Corpus haven't been transformed from word to index yet, please word2idx first")
      require(text2Pos.length == text2Neg.length,
        "text2Corpus contains text2 with different lengths, please shapeSequence first")
      val pairedIndices = text1 ++ text2Pos ++ text1 ++ text2Neg
      val feature = Tensor(pairedIndices, Array(2, text1.length + text2Pos.length))
      val label = Tensor(Array(1.0f, 0.0f), Array(2, 1))
      textFeature(TextFeature.sample) = Sample(feature, label)
      textFeature
    })
    TextSet.rdd(res)
  }

  // Generate RelationLists: each question together with all its answers and labels
  // and transform each RelationList to a TextFeature.
  def fromRelationLists(
      relations: RDD[Relation],
      text1Corpus: TextSet,
      text2Corpus: TextSet): TextSet = {
    require(text1Corpus.isDistributed, "text1Corpus must be a DistributedTextSet")
    require(text2Corpus.isDistributed, "text2Corpus must be a DistributedTextSet")
    val joinedText1 = text1Corpus.toDistributed().rdd.keyBy(_.uri())
      .join(relations.keyBy(_.text1ID)).map(_._2)
    val joinedText2 = text2Corpus.toDistributed().rdd.keyBy(_.uri()).join(
      joinedText1.keyBy(_._2.text2ID))
      .map(x => (x._2._2._1, x._2._1, x._2._2._2.label))
    val joinedLists = joinedText2.groupBy(_._1.uri()).map(_._2.toArray)
    val res = joinedLists.map(x => {
      val text1 = x.head._1
      val text2Array = x.map(_._2)
      val textFeature = TextFeature(null,
        uri = text1.uri() ++ text2Array.map(_.uri()).mkString(""))
      val text1Indices = text1.getIndices
      require(text1Indices != null,
        "text1Corpus haven't been transformed from word to index yet, please word2idx first")
      val text2IndicesArray = text2Array.map(_.getIndices)
      text2IndicesArray.foreach(x => require(x != null,
        "text2Corpus haven't been transformed from word to index yet, please word2idx first"))
      val data = text2IndicesArray.flatMap(text1Indices ++ _)
      val feature = Tensor(data,
        Array(text2Array.length, text1Indices.length + text2IndicesArray.head.length))
      val label = Tensor(x.map(_._3.toFloat), Array(text2Array.length, 1))
      textFeature(TextFeature.sample) = Sample(feature, label)
      textFeature
    })
    TextSet.rdd(res)
  }

  /**
   * Zip word with its corresponding index. Index starts from 1.
   * @param words Array of words, each with its occurrence frequency in descending order.
   * @return WordIndex map.
   */
  def wordsToMap(words: Array[String], existingMap: Map[String, Int] = null): Map[String, Int] = {
    if (existingMap == null) {
      val indexes = Range(1, words.length + 1)
      words.zip(indexes).map{item =>
        (item._1, item._2)}.toMap
    }
    else {
      val resMap = collection.mutable.Map(existingMap.toSeq: _*)
      var i = existingMap.values.max + 1
      for (word <- words) {
        if (!existingMap.contains(word)) {
          resMap(word) = i
          i += 1
        }
      }
      resMap.toMap
    }
  }
}


/**
 * LocalTextSet is comprised of array of TextFeature.
 */
class LocalTextSet(var array: Array[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    array = transformer.apply(array.toIterator).toArray
    this
  }

  override def isLocal: Boolean = true

  override def isDistributed: Boolean = false

  override def toLocal(): LocalTextSet = {
    this
  }

  override def toDistributed(sc: SparkContext, partitionNum: Int = 4): DistributedTextSet = {
    new DistributedTextSet(sc.parallelize(array, partitionNum))
  }

  override def toDataSet: DataSet[Sample[Float]] = {
    DataSet.array(array.map(_[Sample[Float]](TextFeature.sample)))
  }

  override def randomSplit(weights: Array[Double]): Array[TextSet] = {
    throw new UnsupportedOperationException("LocalTextSet doesn't support randomSplit for now")
  }

  override def generateWordIndexMap(
    removeTopN: Int = 0,
    maxWordsNum: Int = -1,
    minFreq: Int = 1,
    existingMap: Map[String, Int] = null): Map[String, Int] = {
    var frequencies = array.flatMap(_.getTokens).filter(_ != "##")  // "##" is the padElement.
      .groupBy(identity).mapValues(_.length).toArray.filter(_._2 >= minFreq)
      .sortBy(- _._2).map(_._1).drop(removeTopN)
    if (maxWordsNum > 0) {
      frequencies = frequencies.take(maxWordsNum)
    }
    val wordIndex = TextSet.wordsToMap(frequencies, existingMap)
    setWordIndex(wordIndex)
    wordIndex
  }
}


/**
 * DistributedTextSet is comprised of RDD of TextFeature.
 */
class DistributedTextSet(var rdd: RDD[TextFeature]) extends TextSet {

  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal: Boolean = false

  override def isDistributed: Boolean = true

  override def toLocal(): LocalTextSet = {
    new LocalTextSet(rdd.collect())
  }

  override def toDistributed(sc: SparkContext = null, partitionNum: Int = 4): DistributedTextSet = {
    this
  }

  override def toDataSet: DataSet[Sample[Float]] = {
    DataSet.rdd(rdd.map(_[Sample[Float]](TextFeature.sample)))
  }

  override def randomSplit(weights: Array[Double]): Array[TextSet] = {
    rdd.randomSplit(weights).map(TextSet.rdd)
  }

  override def generateWordIndexMap(
    removeTopN: Int = 0,
    maxWordsNum: Int = -1,
    minFreq: Int = 1,
    existingMap: Map[String, Int] = null): Map[String, Int] = {
    var frequencies = rdd.flatMap(_.getTokens).filter(_ != "##")  // "##" is the padElement.
      .map(word => (word, 1)).reduceByKey(_ + _).filter(_._2 >= minFreq)
      .sortBy(- _._2).map(_._1).collect().drop(removeTopN)
    if (maxWordsNum > 0) {
      frequencies = frequencies.take(maxWordsNum)
    }
    val wordIndex = TextSet.wordsToMap(frequencies, existingMap)
    setWordIndex(wordIndex)
    wordIndex
  }
}

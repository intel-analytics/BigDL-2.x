/*
 * Copyright 2016 The BigDL Authors.
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
package com.intel.analytics.zoo.pipeline.deepspeech2.pipeline.acoustic

import java.io.{BufferedReader, InputStreamReader}
import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, struct, udf}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class NGramDecoder(override val uid: String, modelPath: String) extends Transformer
  with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this(modelPath: String) = this(Identifiable.randomUID("NGramDecoder"), modelPath)

  val alphabet: Param[String] = new Param[String](this, "alphabet", "alphabet")
  setDefault(alphabet -> "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

  val uttLength = new IntParam(this, "uttLength", "uttLength", ParamValidators.gt(0))
  setDefault(uttLength -> 3000)

  val windowSize = new IntParam(this, "windowSize", "windowSize", ParamValidators.gt(0))
  setDefault(windowSize -> 400)

  val originalSizeCol: Param[String] = new Param[String](this, "originalSizeCol", "originalSizeCol")
  setDefault(originalSizeCol -> "originalSize")

  val n = new IntParam(this, "n", "n", ParamValidators.gt(0))
  setDefault(n -> 2)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group getParam */
  def getAlphabet: String = $(alphabet)

  /** @group setParam */
  def setAlphabet(value: String): this.type = set(alphabet, value)

  /** @group getParam */
  def getUttLength: Int = $(uttLength)

  /** @group setParam */
  def setUttLength(value: Int): this.type = set(uttLength, value)

  /** @group getParam */
  def getWindowSize: Int = $(windowSize)

  /** @group setParam */
  def setWindowSize(value: Int): this.type = set(windowSize, value)

  /** @group getParam */
  def getOriginalSizeCol: String = $(originalSizeCol)

  /** @group setParam */
  def setOriginalSizeCol(value: String): this.type = set(originalSizeCol, value)

  /** @group getParam */
  def getN: Int = $(n)

  /** @group setParam */
  def setN(value: Int): this.type = set(n, value)

  override def transform(dataset: Dataset[_]): DataFrame = {

    val outputSchema = transformSchema(dataset.schema)
    val fs = FileSystem.get(new URI(modelPath), new Configuration())
    val vocab = try {
      val br = new BufferedReader(new InputStreamReader(fs.open(new Path(modelPath, "vocab.txt"))))
      val modelVocab = br.lines().toArray().map(_.asInstanceOf[String].trim.toUpperCase)
        .filter(_.nonEmpty)
      modelVocab
    } finally {
      fs.close()
    }

    val n2List = try {
      val br = new BufferedReader(new InputStreamReader(fs.open(new Path(modelPath, "ngram.txt"))))
      val ngrams = br.lines().toArray().map(_.asInstanceOf[String].trim.toUpperCase)
        .filter(_.nonEmpty)
      ngrams
    } finally {
      fs.close()
    }

    val vocabSet = vocab.toSet
    val n2Model = n2List.map(s => s.split("\\s+"))
      .map(arr => ((1 until $(n)).map(i => arr(i)).mkString(","), (arr($(n)), arr(0).toInt)))
      .groupBy(_._1)
      .map { case (key, values) =>
        (key, values.map(_._2).sortBy(-_._2).map(_._1).toSeq)
      }

    val decoder = new BestPathDecoder($(alphabet), 0, 28)
    val assembleFunc = udf { r: Row =>
      val originSize = r.getInt(0)
      val samples = r.getSeq[Float](1)
      val height = $(alphabet).length
      val width = samples.size / height
      val modelOutput = samples.toArray.grouped(height).toArray
        .slice(0, width * originSize / $(uttLength))
        .transpose
      val so = decoder.getSoftmax(modelOutput)
      val result = decoder.decode(so)
      val words = result.split("\\s+")
      val buffer = new ArrayBuffer[String]()

      var wi = 0
      while (wi < words.length) {
        val word = words(wi)
        val n_1 = $(n) - 1
        val lastword = if (wi >= n_1) {
           (0 until n_1).map(k => n_1 - k).map(k => words(wi - k)).mkString(",")
        } else {
          ""
        }
        val vocabCandidates = vocab
          .map(w => (stringDistance(w, word), w)).filter(_._1 <= 1).map(_._2).toSeq ++ Seq(word)
        val best = if (!vocabSet.contains(word)) {
          val self = search(word, lastword, n2Model, vocabCandidates)
          if (self.nonEmpty)
            self.get
          else
            vocabCandidates.head
        } else {
          if (!n2Model.get(lastword).contains(word)){
            val self = search(word, lastword, n2Model, vocabCandidates)
            if (self.nonEmpty)
              self.get
          }
          word
        }
        buffer += best
        wi += 1
      }

      buffer.mkString(" ").toUpperCase()
    }

    val args = Array($(originalSizeCol), $(inputCol) ).map { c =>
      dataset(c)
    }
    val metadata = new AttributeGroup($(outputCol)).toMetadata()

    dataset.select(col("*"), assembleFunc(struct(args: _*)).as($(outputCol), metadata))
  }

  private def search(word: String, lastword: String, ngramModel: Map[String, Seq[String]],
      vocabCandidates: Seq[String]): Option[String] = {
    val n_1 = $(n) - 1
    val ngramCandidats = ngramModel.getOrElse(lastword, Seq[String]())
    val candidates = ngramCandidats.filter(vocabCandidates.contains(_))
    if (candidates.nonEmpty)
      Some(candidates.head)
    else ngramCandidats.headOption
  }

  private def stringDistance(s1: String, s2: String): Int = {
    def sd(s1: List[Char], s2: List[Char], costs: List[Int]): Int = s2 match {
      case Nil => costs.last
      case c2 :: tail => sd( s1, tail,
        (List(costs.head+1) /: costs.zip(costs.tail).zip(s1))((a,b) => b match {
          case ((rep,ins), chr) =>
            Math.min(Math.min(ins+1, a.head+1 ), rep + (if (chr==c2) 0 else 1) ) :: a
        }).reverse
      )
    }
    sd(s1.toList, s2.toList, (0 to s1.length).toList)
  }

  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), StringType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): NGramDecoder = defaultCopy(extra)
}


object NGramDecoder extends DefaultParamsReadable[NGramDecoder] {

  override def load(path: String): NGramDecoder = super.load(path)
}


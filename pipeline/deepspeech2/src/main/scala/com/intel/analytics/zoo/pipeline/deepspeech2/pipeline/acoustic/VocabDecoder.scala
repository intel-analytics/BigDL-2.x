

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
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

import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, struct, udf}
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class VocabDecoder(override val uid: String, modelPath: String) extends Transformer
  with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this(modelPath: String) = this(Identifiable.randomUID("LanguageDecoder"), modelPath)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val alphabet: Param[String] = new Param[String](this, "alphabet", "alphabet")

  /** @group getParam */
  def getAlphabet: String = $(alphabet)

  /** @group setParam */
  def setAlphabet(value: String): this.type = set(alphabet, value)

  val uttLength = new IntParam(this, "uttLength", "uttLength", ParamValidators.gt(0))
  setDefault(uttLength -> 3000)

  /** @group getParam */
  def getUttLength: Int = $(uttLength)

  /** @group setParam */
  def setUttLength(value: Int): this.type = set(uttLength, value)

  val windowSize = new IntParam(this, "windowSize", "windowSize", ParamValidators.gt(0))
  setDefault(windowSize -> 400)
  /** @group getParam */
  def getWindowSize: Int = $(windowSize)

  /** @group setParam */
  def setWindowSize(value: Int): this.type = set(windowSize, value)

  val originalSizeCol: Param[String] = new Param[String](this, "originalSizeCol", "originalSizeCol")
  setDefault(originalSizeCol -> "originalSize")

  /** @group getParam */
  def getOriginalSizeCol: String = $(originalSizeCol)

  /** @group setParam */
  def setOriginalSizeCol(value: String): this.type = set(originalSizeCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val fs = FileSystem.get(new URI(modelPath), new Configuration())
    val vocab = try {
      val br = new BufferedReader(new InputStreamReader(fs.open(new Path(modelPath, "vocab.txt"))));
      val modelVocab = br.lines().toArray().map(_.asInstanceOf[String].trim).filter(_.nonEmpty).map(_.toUpperCase)
      modelVocab
    } finally {
      fs.close()
    }

    val vocabSet = vocab.toSet
    val decoder = new BestPathDecoder($(alphabet), 0, 28)
    val outputSchema = transformSchema(dataset.schema)
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

      for (word <- words) {
        val best = if (!vocabSet.contains(word)) {
          var minDist = Int.MaxValue
          var minWord = word
          var i = 0
          val maxLength = vocab.length
          var found = false
          while (i < maxLength && !found) {
            val w = vocab(i)
            val dist = stringDistance(w, word)
            if (dist < minDist) {
              minDist = dist
              minWord = w
            }
            if(dist <= 1) found = true
            i += 1
          }
          minWord
        } else {
          word
        }
        buffer += best
      }

      buffer.mkString(" ").toUpperCase()
    }

    val args = Array($(originalSizeCol), $(inputCol) ).map { c => dataset(c) }
    val metadata = new AttributeGroup($(outputCol)).toMetadata()

    dataset.select(col("*"), assembleFunc(struct(args: _*)).as($(outputCol), metadata))
  }

  private def stringDistance(s1: String, s2: String): Int = {
    def sd(s1: List[Char], s2: List[Char], costs: List[Int]): Int = s2 match {
      case Nil => costs.last
      case c2 :: tail => sd( s1, tail,
        (List(costs.head+1) /: costs.zip(costs.tail).zip(s1))((a,b) => b match {
          case ((rep,ins), chr) => Math.min( Math.min( ins+1, a.head+1 ), rep + (if (chr==c2) 0 else 1) ) :: a
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

  override def copy(extra: ParamMap): VocabDecoder = defaultCopy(extra)
}


object VocabDecoder extends DefaultParamsReadable[VocabDecoder] {
  override def load(path: String): VocabDecoder = super.load(path)
}


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
package com.intel.analytics.deepspeech2.pipeline.acoustic

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * vectorize the transcript with the alphabet, and extend it to a specific length maxTscrptLen.
 */
class TranscriptVectorizer ( override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("TranscriptVectorizer"))

  val alphabet: Param[String] = new Param[String](this, "alphabet", "alphabet")

  val maxTscrptLen: IntParam = new IntParam(this, "maxTscrptLen", "maxTscrptLen")

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group getParam */
  def getAlphabet: String = $(alphabet)

  /** @group setParam */
  def setAlphabet(value: String): this.type = set(alphabet, value)

  /** @group getParam */
  def getMaxTscrptLen: Int = $(maxTscrptLen)

  /** @group setParam */
  def setMaxTscrptLen(value: Int): this.type = set(maxTscrptLen, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema)
    val dict = $(alphabet).toCharArray.zipWithIndex.toMap

    val convert = udf { transcript: String =>
      val arr = transcript.toCharArray.map(c => dict(c).toFloat)
      require($(maxTscrptLen) > arr.length)
      val vec = arr ++ Array.fill($(maxTscrptLen) - arr.length)(0.0F)
      vec
    }
    dataset.withColumn($(outputCol), convert(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new ArrayType(IntegerType, false), false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): TranscriptVectorizer = defaultCopy(extra)
}


object TranscriptVectorizer extends DefaultParamsReadable[TranscriptVectorizer] {
  override def load(path: String): TranscriptVectorizer = super.load(path)
}


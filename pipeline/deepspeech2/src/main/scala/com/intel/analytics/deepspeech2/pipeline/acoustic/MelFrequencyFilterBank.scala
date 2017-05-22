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

package com.intel.analytics.deepspeech2.pipeline.acoustic

import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import breeze.numerics
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


class MelFrequencyFilterBank ( override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {


  def this() = this(Identifiable.randomUID("MelFrequencyFilterBank"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)


  val windowSize = new IntParam(this, "windowSize", "windowSize",
    ParamValidators.gt(0))

  val sampleFreq = new IntParam(this, "sampleFreq", "sampleFreq",
    ParamValidators.gt(0))

  /** @group getParam */
  def getWindowSize: Int = $(windowSize)

  /** @group setParam */
  def setWindowSize(value: Int): this.type = set(windowSize, value)

  val numFilters = new IntParam(this, "numFilters", "numFilters",
    ParamValidators.gt(0))
  /** @group getParam */
  def getNumFilters: Int = $(numFilters)

  /** @group setParam */
  def setNumFilters(value: Int): this.type = set(numFilters, value)

  val uttLength = new IntParam(this, "uttLength", "uttLength",
    ParamValidators.gt(0))

  /** @group getParam */
  def getUttLength: Int = $(uttLength)

  /** @group setParam */
  def setUttLength(value: Int): this.type = set(uttLength, value)

  /** @group getParam */
  def getSampleFreq: Int = $(sampleFreq)

  /** @group setParam */
  def setSampleFreq(value: Int): this.type = set(sampleFreq, value)
  setDefault(windowSize -> 400, numFilters -> 13, sampleFreq -> 16000)
  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema)
    val filterbank = createFilterBank($(numFilters), $(windowSize), $(sampleFreq))

    val convert = udf { (samples: mutable.WrappedArray[Float]) =>
      val count = samples.size / ($(windowSize) / 2 + 1)
      val specgram = new DenseMatrix(count, ($(windowSize) / 2 + 1), samples.toArray)
      val mm = specgram :* specgram
      mm :*= 1.0F / $(windowSize)

      val mm2 = mm * filterbank
      val mm3 = (numerics.log(mm2))

      val mm6 = if (isDefined(uttLength)) {
        if($(uttLength) > mm3.rows) {
          val mm4 = DenseMatrix.zeros[Float](($(uttLength) - mm3.rows), mm3.cols)
          DenseMatrix.vertcat(mm3, mm4)
        } else {
          mm3(0 to $(uttLength), ::)
        }
      } else {
       mm3
      }

      mm6.toArray
    }

    dataset.withColumn($(outputCol), convert(col($(inputCol))))
  }

  private def createFilterBank(numFilter: Int, fftsz: Int, sampleFreqHz: Int): DenseMatrix[Float] = {
    val minMelFreq = hz_to_mel(0.0)
    val maxMelFreq = hz_to_mel(sampleFreqHz / 2.0)
    val melFreqDelta = (maxMelFreq - minMelFreq) / (numFilter + 1)

    val bins = new ArrayBuffer[Int]()
    val scaleby = (1 + fftsz).toDouble / sampleFreqHz
    for( j <- 0 to 1) {
      bins += Math.floor(scaleby * mel_to_hz(minMelFreq + j * melFreqDelta)).toInt
    }

    val numFreqs = fftsz / 2 + 1
    val fbank = DenseMatrix.zeros[Float](numFreqs, numFilter)

    for (j <- 0 until numFilter) {
      bins += Math.floor(scaleby * mel_to_hz(minMelFreq + (j + 2) * melFreqDelta)).toInt
      var i = bins(j)
      while( i < bins(j + 1)) {
        fbank(i, j) = (i - bins(j)).toFloat / (1.0F * (bins(j + 1) - bins(j)))
        i += 1
      }

      i = bins(j + 1)
      while ( i < bins(j + 2)) {
        fbank(i, j) =(bins(j + 2) - i) / (1.0F * (bins(j + 2) - bins(j + 1)))
        i += 1
      }
    }
    fbank
  }

  private def hz_to_mel(freq_hz: Double): Double = {
    2595 * math.log10(1 + freq_hz / 700.0)
  }

  private def mel_to_hz(freq_mel: Double): Double = {
    700 * (math.pow(10, freq_mel / 2595.0) - 1)
  }



  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), schema($(inputCol)).dataType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): MelFrequencyFilterBank = defaultCopy(extra)
}


object MelFrequencyFilterBank extends DefaultParamsReadable[MelFrequencyFilterBank] {

  override def load(path: String): MelFrequencyFilterBank = super.load(path)
}

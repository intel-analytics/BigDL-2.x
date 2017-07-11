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

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.math.Complex
import breeze.signal.fourierTr
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable

class DFTSpecgram ( override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("DFTSpecgram"))

  val windowSize = new IntParam(this, "windowSize", "windowSize", ParamValidators.gt(0))
  setDefault(windowSize -> 400)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group getParam */
  def getWindowSize: Int = $(windowSize)

  /** @group setParam */
  def setWindowSize(value: Int): this.type = set(windowSize, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema)
    val convert = udf { (samples: mutable.WrappedArray[Float]) =>
      val count = samples.size / $(windowSize)
      val m = new BDM[Float]($(windowSize), count, samples.toArray).t
      val compx = m.map(real => Complex(real, 0))
      for (row <- 0 until m.rows) {
        val arr = compx(row, ::)
        val complexArr = arr.inner
        val dft = fourierTr(complexArr)
        val spec = dft.map(c => Math.sqrt(c.real * c.real + c.im() * c.im()).toFloat)
        m(row, ::) := spec.t
      }
      val specgram = m (::, 0 until ($(windowSize) / 2 + 1))
      specgram.toArray
    }
    dataset.withColumn($(outputCol), convert(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), schema($(inputCol)).dataType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): DFTSpecgram = defaultCopy(extra)
}


object DFTSpecgram extends DefaultParamsReadable[DFTSpecgram] {

  override def load(path: String): DFTSpecgram = super.load(path)

}


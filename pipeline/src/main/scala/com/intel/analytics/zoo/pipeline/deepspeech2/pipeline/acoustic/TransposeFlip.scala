
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

import breeze.linalg.{fliplr, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.{ParamMap, _}
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.collection.mutable


class TransposeFlip ( override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("TransposeFlip"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)


  val numFilters = new IntParam(this, "numFilters", "numFilters", ParamValidators.gt(0))

  setDefault(numFilters -> 13)

  /** @group getParam */
  def getNumFilters: Int = $(numFilters)

  /** @group setParam */
  def setNumFilters(value: Int): this.type = set(numFilters, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema)

    val convert = udf { (samples: mutable.WrappedArray[Float]) =>
      val count = samples.size / $(numFilters)
      val ss = SparkSession.builder().getOrCreate()
      val min = 0
      val max = 255
      val oldMax = samples.max
      val oldMin = samples.min
      val oldRange = oldMax - oldMin

      val normArray = samples.map { d =>
        if (!d.isNaN) {
          val raw = if (oldRange != 0) (d - oldMin) / oldRange else 0.5F
          raw * 255
        } else {
          d
        }
      }

      val m = new BDM[Float](count, $(numFilters), normArray.toArray)
      val result = fliplr(m).t
      result.data.map(d => Math.round(d).toFloat)
    }

    dataset.withColumn($(outputCol), convert(col($(inputCol))))
  }


  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), schema($(inputCol)).dataType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): TransposeFlip = defaultCopy(extra)
}


object TransposeFlip extends DefaultParamsReadable[TransposeFlip] {

  override def load(path: String): TransposeFlip = super.load(path)
}




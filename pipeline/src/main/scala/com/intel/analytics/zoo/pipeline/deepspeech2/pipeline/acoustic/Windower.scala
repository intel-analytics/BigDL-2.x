
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

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable


class Windower ( override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {


  def this() = this(Identifiable.randomUID("Windower"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)


  val windowSize = new IntParam(this, "windowSize", "windowSize",
    ParamValidators.gt(0))

  val windowShift = new IntParam(this, "windowShift", "windowShift")

  setDefault(windowSize -> 400, windowShift -> 160)

  /** @group getParam */
  def getWindowSize: Int = $(windowSize)

  /** @group setParam */
  def setWindowSize(value: Int): this.type = set(windowSize, value)

  /** @group getParam */
  def getWindowShift: Int = $(windowShift)

  /** @group setParam */
  def setWindowShift(value: Int): this.type = set(windowShift, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema)
    val reScale = udf { (samples: mutable.WrappedArray[Float]) =>
      val alpha = 0.5
      val hanningWindow = (0 until $(windowSize))
        .map(i => 1- alpha - alpha * Math.cos(2 * Math.PI * i / ($(windowSize) - 1.0))).map(_.toFloat)
      val rawFrames = samples.sliding($(windowSize),  $(windowShift))
      val frames = rawFrames.map(f => f.zip(hanningWindow).map { case(a, b) => a * b })
      val arr = frames.flatMap( a => a.iterator).toArray
      arr
    }

    dataset.withColumn($(outputCol), reScale(col($(inputCol))))
  }


  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), schema($(inputCol)).dataType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): Windower = defaultCopy(extra)
}


object Windower extends DefaultParamsReadable[Windower] {

  override def load(path: String): Windower = super.load(path)
}

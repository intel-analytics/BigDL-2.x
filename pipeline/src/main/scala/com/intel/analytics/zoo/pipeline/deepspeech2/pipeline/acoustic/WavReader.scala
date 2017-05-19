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

import java.io.{ByteArrayInputStream, File}
import java.net.URI
import java.util
import javax.sound.sampled.AudioSystem

import com.sun.media.sound.WaveFileReader
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}


class WavReader ( override val uid: String, host: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this(host: String) = this(Identifiable.randomUID("FlacReader"), host)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema)

    val reScale = udf { path: String =>
      WavReader.pathToSamples(path)
    }
    dataset.withColumn($(outputCol), reScale(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new ArrayType(FloatType, false), false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): WavReader = defaultCopy(extra)
}


object WavReader extends DefaultParamsReadable[WavReader] {

  override def load(path: String): WavReader = super.load(path)

  def pathToSamples(path: String): Array[Float] = {
    val src: Path = new Path(path)
    val fs = src.getFileSystem(new Configuration())
    val is = fs.open(src)
    val audioInputStream = new WaveFileReader().getAudioInputStream(is)
    val bytesArray = new Array[Byte](audioInputStream.available())
    audioInputStream.read(bytesArray)
    val shorts = bytesArray.grouped(2).map { case Array(lo, hi) =>
      (((hi & 0xFF) << 8) | (lo & 0xFF)).asInstanceOf[Short]
    }
    shorts.map(_.toFloat).toArray
  }
}

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

package org.apache.spark.ml.feature

import scala.collection.mutable.ArrayBuffer

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.kc7bfi.jflac.FLACDecoder


class FlacReader ( override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("FlacReader"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema)
    val reScale = udf { path: String =>
      FlacReader.pathToSamples(path)
    }
    dataset.withColumn($(outputCol), reScale(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): FlacReader = defaultCopy(extra)
}


object FlacReader extends DefaultParamsReadable[FlacReader] {

  override def load(path: String): FlacReader = super.load(path)

  val pathToSamples: String => Array[Float] = (path: String) => {

    val src: Path = new Path(path)
    val conf = new Configuration()
    conf.setBoolean("fs.hdfs.impl.disable.cache", true)
    val fs = src.getFileSystem(conf)
    val is = fs.open(src)

    val decoder = new FLACDecoder(is)
    decoder.readMetadata()
    val buf = new ArrayBuffer[Short]()
    var frame = decoder.readNextFrame()
    while(frame != null) {
      val pcm = decoder.decodeFrame(frame, null)
      val shorts = pcm.getData.take(pcm.getLen).grouped(2).map { case Array(lo, hi) =>
        (((hi & 0xFF) << 8) | (lo & 0xFF)).asInstanceOf[Short]
      }
      buf ++= shorts
      frame = decoder.readNextFrame()
    }
    fs.close()
    buf.map(_.toFloat).toArray
  }
}


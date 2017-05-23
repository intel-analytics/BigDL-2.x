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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable

class DeepSpeech2ModelTransformer ( override val uid: String, modelPath: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this(modelPath: String) = this(Identifiable.randomUID("DeepSpeech2ModelTransformer"), modelPath)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val numFilters = new IntParam(this, "numFilters", "numFilters", ParamValidators.gt(0))

  /** @group getParam */
  def getNumFilters: Int = $(numFilters)

  /** @group setParam */
  def setNumFilters(value: Int): this.type = set(numFilters, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val model = DeepSpeech2ModelLoader.loadModel(modelPath)
    val outputSchema = transformSchema(dataset.schema)
    val height = $(numFilters)
    val reScale = udf { (samples: mutable.WrappedArray[Float]) =>
      val width = samples.size / height
      val input = Tensor[Float](Storage(samples.toArray), 1, Array(1, 1, height, width))
      val output = model.forward(input).toTensor[Float].transpose(2, 3)
      output.storage().toArray
    }

    dataset.withColumn($(outputCol), reScale(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), schema($(inputCol)).dataType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): DeepSpeech2ModelTransformer = defaultCopy(extra)
}


object DeepSpeech2ModelTransformer extends DefaultParamsReadable[DeepSpeech2ModelTransformer] {
  override def load(path: String): DeepSpeech2ModelTransformer = {
    super.load(path)
  }
}

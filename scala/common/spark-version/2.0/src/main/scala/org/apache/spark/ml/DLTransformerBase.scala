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
package org.apache.spark.ml

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}

/**
 * A wrapper for org.apache.spark.ml.Transformer.
 * Extends MlTransformer and override process to gain compatibility with
 * both spark 1.5 and spark 2.0.
 */
private[ml] abstract class DLTransformerBase[M <: DLTransformerBase[M]]
  extends Model[M] with DLParams {

  /**
   * convert feature columns(MLlib Vectors or Array) to Seq format
   */
  protected def internalTransform(featureData: RDD[Seq[AnyVal]], dataset: DataFrame): DataFrame

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    internalTransform(toArrayType(dataset.toDF()), dataset.toDF())
  }

  /**
   * convert feature columns to Seq format
   */
  protected def toArrayType(dataset: DataFrame): RDD[Seq[AnyVal]] = {

    val featureType = dataset.schema($(featuresCol)).dataType
    val featureColIndex = dataset.schema.fieldIndex($(featuresCol))

    dataset.rdd.map { row =>
      val features = supportedTypesToSeq(row, featureType, featureColIndex)
      features
    }
  }

  override def copy(extra: ParamMap): M = defaultCopy(extra)
}

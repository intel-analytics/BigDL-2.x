/*
 * Copyright 2018 Analytics Zoo Authors.
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
package org.apache.spark.ml.adapter

/**
 * Handle different Vector types in Spark 1.5/1.6
 */
trait VectorToSeq {

  def vectorToArray(v: Any): Array[Double] = {
    v match {
      case mllibdv: org.apache.spark.mllib.linalg.DenseVector => mllibdv.toArray
      case mllibsv: org.apache.spark.mllib.linalg.SparseVector => mllibsv.toArray
      case _ => throw new IllegalArgumentException(s"$v is not a supported vector type.")
    }
  }
}

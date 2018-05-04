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
package com.intel.analytics.zoo.pipeline.nnframes.transformers

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.pipeline.nnframes.NNImageSchema
import org.apache.spark.sql.Row

import scala.reflect.ClassTag

/**
 * a Transformer that converts a Spark Row to a BigDL ImageFeature.
 */
class RowToImageFeature[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Transformer[Row, ImageFeature] {

  override def apply(prev: Iterator[Row]): Iterator[ImageFeature] = {
    prev.map { row =>
      NNImageSchema.row2IMF(row)
    }
  }
}

object RowToImageFeature {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): RowToImageFeature[T] =
    new RowToImageFeature[T]()
}

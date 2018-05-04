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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.mllib.linalg.Vector

import scala.reflect.ClassTag

/**
 * a Transformer that converts MLlib Vector to a Tensor.
 * @param size dimensions of target Tensor.
 */
class MLlibVectorToTensor[T: ClassTag](size: Array[Int])(implicit ev: TensorNumeric[T])
  extends Transformer[Vector, Tensor[T]] {

  override def apply(prev: Iterator[Vector]): Iterator[Tensor[T]] = {
    prev.map(a => Tensor(a.toArray.map(ev.fromType(_)), size))
  }
}

object MLlibVectorToTensor {
  def apply[T: ClassTag](size: Array[Int])(implicit ev: TensorNumeric[T]): MLlibVectorToTensor[T] =
    new MLlibVectorToTensor[T](size)
}

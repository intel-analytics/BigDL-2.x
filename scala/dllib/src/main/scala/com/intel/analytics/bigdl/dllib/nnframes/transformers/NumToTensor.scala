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

import scala.reflect.ClassTag

class NumToTensor[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Transformer[AnyVal, Tensor[T]] {

  override def apply(prev: Iterator[AnyVal]): Iterator[Tensor[T]] = {
    prev.map { f =>
      val feature = f match {
        case dd: Double => ev.fromType(f.asInstanceOf[Double])
        case ff: Float => ev.fromType(f.asInstanceOf[Float])
        case _ => throw new IllegalArgumentException("NumToTensor only supports Float and Double")
      }
      Tensor(Array(feature), Array(1))
    }
  }
}

object NumToTensor {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]) = new NumToTensor[T]()
}

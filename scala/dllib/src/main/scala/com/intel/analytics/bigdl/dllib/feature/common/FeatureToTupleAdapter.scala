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
package com.intel.analytics.zoo.feature.common

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class TupleToFeatureAdapter[F, L, T](
    featureTransformer: Transformer[F, Sample[T]]
  )(implicit ev: TensorNumeric[T]) extends Preprocessing[(F, Option[L]), Sample[T]] {

  override def apply(prev: Iterator[(F, Option[L])]): Iterator[Sample[T]] = {
    featureTransformer.apply(prev.map(_._1))
  }
}

object TupleToFeatureAdapter {
  def apply[F, L, T: ClassTag](
      featureTransformer: Transformer[F, Sample[T]]
    )(implicit ev: TensorNumeric[T]): TupleToFeatureAdapter[F, L, T] =
    new TupleToFeatureAdapter(featureTransformer)
}

class ToTuple extends Preprocessing[Any, (Any, Option[Any])] {
  override def apply(prev: Iterator[Any]): Iterator[(Any, Option[Any])] = {
    prev.map(f => (f, None))
  }
}

object ToTuple {
  def apply(): ToTuple = new ToTuple
}

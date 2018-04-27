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

import com.intel.analytics.bigdl.dataset.{Sample, SampleToMiniBatch, Transformer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[nnframes] class FeatureLabelTransformer[F, L, T: ClassTag] (
    featureTransfomer: Transformer[F, Tensor[T]],
    labelTransformer: Transformer[L, Tensor[T]])(implicit ev: TensorNumeric[T])
  extends Transformer[(F, L), Sample[T]] {

  override def apply(prev: Iterator[(F, L)]): Iterator[Sample[T]] = {
    prev.map { case (feature, label ) =>
      val featureTensor = featureTransfomer(Iterator(feature)).next()
      val labelTensor = labelTransformer(Iterator(label)).next()
      Sample[T](featureTensor, labelTensor)
    }
  }
}

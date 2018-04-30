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

/**
 * construct a Transformer that convert (Feature, Label) tuple to a Sample.
 * The returned Transformer is robust for the case label = null, in which the 
 * Sample is derived from Feature only.
 * @param featureTransfomer transformer for feature, transform F to Tensor[T]
 * @param labelTransformer transformer for label, transform L to Tensor[T]
 * @tparam F data type from feature column, E.g. Array[_] or Vector
 * @tparam L data type from label column, E.g. Float, Double, Array[_] or Vector
 */
class FeatureLabelTransformer[F, L, T: ClassTag] (
    featureTransfomer: Transformer[F, Tensor[T]],
    labelTransformer: Transformer[L, Tensor[T]])(implicit ev: TensorNumeric[T])
  extends Transformer[(F, L), Sample[T]] {

  override def apply(prev: Iterator[(F, L)]): Iterator[Sample[T]] = {
    prev.map { case (feature, label ) =>
      val featureTensor = featureTransfomer(Iterator(feature)).next()
      if (label != null) {
        val labelTensor = labelTransformer(Iterator(label)).next()
        Sample[T](featureTensor, labelTensor)
      } else {
        Sample[T](featureTensor)
      }
    }
  }
}


object FeatureLabelTransformer {
  def apply[F, L, T: ClassTag](
      featureTransfomer: Transformer[F, Tensor[T]],
      labelTransformer: Transformer[L, Tensor[T]]
    )(implicit ev: TensorNumeric[T]) =
    new FeatureLabelTransformer(featureTransfomer, labelTransformer)
}

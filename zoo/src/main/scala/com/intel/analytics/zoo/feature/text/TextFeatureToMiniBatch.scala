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

package com.intel.analytics.zoo.feature.text

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Transform Sample of a TextFeature to MiniBatch, the result of which
 * can be directly fed into the optimizer for training and validation.
 */
class TextFeatureToMiniBatch[T: ClassTag](
    batchSize: Int,
    featurePaddingParam: Option[PaddingParam[T]] = None,
    labelPaddingParam: Option[PaddingParam[T]] = None,
    partitionNum: Option[Int] = None,
    sampleKey: String = TextFeature.sample)(implicit ev: TensorNumeric[T])
  extends Transformer[TextFeature, MiniBatch[T]] {

  val toBatch: SampleToMiniBatch[T] = SampleToMiniBatch[T](
    batchSize, featurePaddingParam, labelPaddingParam, partitionNum)

  override def apply(prev: Iterator[TextFeature]): Iterator[MiniBatch[T]] = {
    toBatch(prev.map(_[Sample[T]](sampleKey)))
  }
}

object TextFeatureToMiniBatch {
  def apply[T: ClassTag](
      batchSize: Int,
      featurePaddingParam: Option[PaddingParam[T]] = None,
      labelPaddingParam: Option[PaddingParam[T]] = None,
      partitionNum: Option[Int] = None,
      sampleKey: String = TextFeature.sample)
      (implicit ev: TensorNumeric[T]): TextFeatureToMiniBatch[T] =
    new TextFeatureToMiniBatch[T](batchSize, featurePaddingParam,
      labelPaddingParam, partitionNum, sampleKey)
}

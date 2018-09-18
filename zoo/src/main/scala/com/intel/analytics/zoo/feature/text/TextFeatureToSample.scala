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

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Transform indexedTokens and label (if any) of a TextFeature to a BigDL Sample.
 * Input key: TextFeature.indexedTokens and TextFeature.label (if any)
 * Output key: TextFeature.sample
 */
class TextFeatureToSample[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TextTransformer {

  override def transform(feature: TextFeature): TextFeature = {
    require(feature.contains(TextFeature.indexedTokens), "TextFeature doesn't contain indexTokens" +
      " yet. Please use WordIndexer to transform tokens to indexedTokens first")
    val indexedTokens = feature[Array[Int]](TextFeature.indexedTokens)
    val input = Tensor[T](data = indexedTokens.map(ev.fromType[Int]),
      shape = Array(indexedTokens.length))
    val sample = if (feature.hasLabel) {
      Sample[T](input, ev.fromType[Int](feature.getLabel))
    }
    else {
      Sample[T](input)
    }
    feature(TextFeature.sample) = sample
    feature
  }
}

object TextFeatureToSample {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): TextFeatureToSample[T] = {
    new TextFeatureToSample[T]()
  }
}

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

class TextSetToSample extends TextTransformer {

  override def transform(feature: TextFeature): TextFeature = {
    val tokens = feature.apply[Array[Int]](TextFeature.indexedTokens)
    // TODO: T instead of Float
    val tensor = Tensor[Float](data = tokens.map(_.toFloat), shape = Array(tokens.length))
    // TODO: handle no label case
    val label = feature.apply[Option[Float]](TextFeature.label).get
    val sample = Sample(tensor, label)
    feature.update(TextFeature.sample, sample)
    feature
  }
}

object TextSetToSample {
  def apply(): TextSetToSample = {
    new TextSetToSample()
  }
}

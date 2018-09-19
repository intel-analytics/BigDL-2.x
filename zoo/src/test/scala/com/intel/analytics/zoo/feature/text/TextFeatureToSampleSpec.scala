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
import org.scalatest.{FlatSpec, Matchers}

class TextFeatureToSampleSpec extends FlatSpec with Matchers  {
  val text = "hello my friend, please annotate my text"
  val feature = TextFeature(text, label = 1)
  feature(TextFeature.indexedTokens) = Array(1, 2, 3, 4, 5, 2, 6)

  "TextFeatureToSample" should "work properly" in {
    val toSample = TextFeatureToSample[Float]()
    val transformed = toSample.transform(feature)
    val sample = transformed[Sample[Float]](TextFeature.sample)
    require(sample.getData().sameElements(Array(1.0f, 2.0f,
      3.0f, 4.0f, 5.0f, 2.0f, 6.0f, 1.0f)))
  }
}

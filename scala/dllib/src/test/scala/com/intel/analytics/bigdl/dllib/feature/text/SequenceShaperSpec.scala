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

import org.scalatest.{FlatSpec, Matchers}

class SequenceShaperSpec extends FlatSpec with Matchers {
  def genFeature(): TextFeature = {
    val text = "please annotate my text"
    val feature = TextFeature(text, label = 0)
    feature(TextFeature.indexedTokens) = Array(1.0f, 2.0f, 3.0f, 4.0f)
    feature
  }

  "SequenceShaper trun pre for indices" should "work properly" in {
    val transformer = SequenceShaper(len = 2)
    val transformed = transformer.transform(genFeature())
    require(transformed.getIndices.sameElements(Array(3.0f, 4.0f)))
  }

  "SequenceShaper trun post for indices" should "work properly" in {
    val transformer = SequenceShaper(len = 3, truncMode = TruncMode.post)
    val transformed = transformer.transform(genFeature())
    require(transformed.getIndices.sameElements(Array(1.0f, 2.0f, 3.0f)))
  }

  "SequenceShaper pad for indices" should "work properly" in {
    val transformer = SequenceShaper(len = 7)
    val transformed = transformer.transform(genFeature())
    require(transformed.getIndices.sameElements(Array(1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f)))
  }
}

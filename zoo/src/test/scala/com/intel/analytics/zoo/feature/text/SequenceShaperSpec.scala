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
    feature(TextFeature.tokens) = Array("please", "annotate", "my", "text")
    feature(TextFeature.indexedTokens) = Array(1, 2, 3, 4)
    feature
  }

  "SequenceShaper trun pre for tokens" should "work properly" in {
    val transformer = SequenceShaper(len = 2, inputKey = TextFeature.tokens)
    val transformed = transformer.transform(genFeature())
    require(transformed.getTokens.sameElements(Array("my", "text")))
  }

  "SequenceShaper trun post for indexedTokens" should "work properly" in {
    val transformer = SequenceShaper(len = 3, truncMode = TruncMode.post)
    val transformed = transformer.transform(genFeature())
    require(transformed[Array[Int]](TextFeature.indexedTokens).sameElements(Array(1, 2, 3)))
  }

  "SequenceShaper pad for indexedTokens" should "work properly" in {
    val transformer = SequenceShaper(len = 6)
    val transformed = transformer.transform(genFeature())
    require(transformed[Array[Int]](TextFeature.indexedTokens).sameElements(
      Array(1, 2, 3, 4, 0, 0)))
  }

  "SequenceShaper pad for tokens" should "work properly" in {
    val transformer = SequenceShaper(len = 7, inputKey = TextFeature.tokens,
      padElement = "##")
    val transformed = transformer.transform(genFeature())
    require(transformed.getTokens.sameElements(
      Array("please", "annotate", "my", "text", "##", "##", "##")))
  }
}

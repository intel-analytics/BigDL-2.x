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

import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class TokenizerSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val text = "Hello my friend, please annotate my text"
  val feature = TextFeature(text)

  private def genFeature(): TextFeature = {
    TextFeature(text, label = 0)
  }

  "Tokenizer and Normalizer" should "work properly" in {
    val tokenizer = Tokenizer()
    val tokenized = tokenizer.transform(feature)
    require(tokenized.keys().contains("tokens"))
    require(tokenized[Array[String]](TextFeature.tokens).sameElements(Array("Hello", "my",
    "friend,", "please", "annotate", "my", "text")))
    val normalizer = Normalizer()
    val normalized = normalizer.transform(tokenized)
    require(normalized.keys().contains("tokens"))
    require(normalized[Array[String]](TextFeature.tokens).sameElements(Array("hello", "my",
      "friend", "please", "annotate", "my", "text")))
  }
}

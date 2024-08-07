/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.dllib.feature.text

import com.intel.analytics.bigdl.dllib.utils.TestUtils
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class WordIndexerSpec extends FlatSpec with Matchers with BeforeAndAfter {
  val text = "hello my friend, please annotate my text"
  val feature = TextFeature(text)
  feature(TextFeature.tokens) = Array("hello", "my", "friend", "please",
    "annotate", "my", "text")

  "WordIndexer" should "work properly" in {
    val wordIndex = Map("friend" -> 1, "my" -> 2, "annotate" -> 3, "text" -> 4)
    val wordIndexer = WordIndexer(wordIndex)
    val transformed = wordIndexer.transform(feature)
    TestUtils.conditionFailTest(
      transformed.getIndices.sameElements(Array(2.0f, 1.0f, 3.0f, 2.0f, 4.0f)))
  }
}

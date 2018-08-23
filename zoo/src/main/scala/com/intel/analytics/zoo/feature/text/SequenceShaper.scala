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

class SequenceShaper(
    len: Int,
    truncMode: String = "pre",
    key: String = TextFeature.indexedTokens) extends TextTransformer {

  require(truncMode == "pre" || truncMode == "post", "truncMode can only be pre or 'post")

  override def transform(feature: TextFeature): TextFeature = {
    // TODO: support key=tokens?
    require(key == TextFeature.indexedTokens)
    val tokens = feature.apply[Array[Int]](key)
    val paddedTokens = if (tokens.length > len) {
      if (truncMode == "pre") {
        tokens.slice(tokens.length - len, tokens.length)
      } else {
        tokens.slice(0, len)
      }
    } else {
        tokens ++ Array.fill[Int](len - tokens.length)(0)
    }
    feature.update(key, paddedTokens)
    feature
  }
}

object SequenceShaper {
  def apply(
     len: Int,
     trunc: String = "pre",
     key: String = TextFeature.indexedTokens): SequenceShaper = {
    new SequenceShaper(len, trunc, key)
  }
}

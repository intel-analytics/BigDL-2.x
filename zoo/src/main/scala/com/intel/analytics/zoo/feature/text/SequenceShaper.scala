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

/**
 * Shape sequence of tokens or indexedTokens to fixed length.
 * If the original sequence is longer than the target length, it will be truncated from
 * the beginning or the end.
 * If the original sequence is shorter than the target length, it will be padded from the
 * beginning or the end.
 *
 * @param len The target length.
 * @param mode Truncation or padding mode. Either 'pre' or 'post'. Default is 'pre'.
 *             If 'pre', the sequence will be truncated from or padded to the beginning.
 *             If 'post', the sequence will be truncated from or padded to the end.
 * @param inputKey The key for the sequence. Either 'tokens' or 'indexedTokens'.
 *                 The output key would be the same. Namely the original sequence will be
 *                 replaced by the shaped sequence.
 *                 Default is 'indexedTokens'.
 * @param padElement The element to be padded to the sequence if its length is smaller than
 *                   the original length.
 *                   It should be a string if inputKey is 'tokens'.
 *                   It should be an integer if inputKey is 'indexedTokens'.
 *                   Default is 0 for 'indexedTokens'.
 */
class SequenceShaper(
    val len: Int,
    val mode: String = "pre",
    val inputKey: String = TextFeature.indexedTokens,
    val padElement: Any = 0) extends TextTransformer {

  require(len > 0, "len should be positive")
  require(mode == "pre" || mode == "post", "truncMode can only be pre or 'post")
  require(inputKey == TextFeature.indexedTokens || inputKey == TextFeature.tokens,
  "inputKey should be tokens or indexedTokens")

  override def transform(feature: TextFeature): TextFeature = {
    require(feature.contains(inputKey), s"TextFeature doesn't have key: $inputKey")
    val tokens = feature.apply[Array[_]](inputKey)
    val paddedTokens = if (tokens.length > len) {
      if (mode == "pre") {
        tokens.slice(tokens.length - len, tokens.length)
      } else {
        tokens.slice(0, len)
      }
    } else {
      if (mode == "pre") {
        if (inputKey == TextFeature.indexedTokens) {
          require(padElement.isInstanceOf[Int], "padElement should be an int for indexedTokens")
          Array.fill[Int](len - tokens.length)(padElement.asInstanceOf[Int]) ++ tokens
        }
        else {
          require(padElement.isInstanceOf[String], "padElement should be a string for tokens")
          Array.fill[String](len - tokens.length)(padElement.asInstanceOf[String]) ++ tokens
        }
      }
      else {
        if (inputKey == TextFeature.indexedTokens) {
          require(padElement.isInstanceOf[Int], "padElement should be an int for indexedTokens")
          tokens ++ Array.fill[Int](len - tokens.length)(padElement.asInstanceOf[Int])
        }
        else {
          require(padElement.isInstanceOf[String], "padElement should be a string for tokens")
          tokens ++ Array.fill[String](len - tokens.length)(padElement.asInstanceOf[String])
        }
      }
    }
    if (inputKey == TextFeature.indexedTokens) {
      feature(inputKey) = paddedTokens.map(_.asInstanceOf[Int])
    }
    else {
      feature(inputKey) = paddedTokens.map(_.asInstanceOf[String])
    }
    feature
  }
}

object SequenceShaper {
  def apply(
      len: Int,
      mode: String = "pre",
      inputKey: String = TextFeature.indexedTokens,
      padElement: Any = 0): SequenceShaper = {
    new SequenceShaper(len, mode, inputKey, padElement)
  }
}

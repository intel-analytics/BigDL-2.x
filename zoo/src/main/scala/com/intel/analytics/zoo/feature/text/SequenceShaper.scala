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

import com.intel.analytics.zoo.feature.text.TruncMode.TruncMode

/**
 * Shape the sequence of indexedTokens or tokens to a fixed length.
 * If the original sequence is longer than the target length, it will be truncated from
 * the beginning or the end.
 * If the original sequence is shorter than the target length, it will be padded to the end.
 *
 * @param len The target length.
 * @param truncMode Truncation mode. Either 'pre' or 'post'. Default is 'pre'.
 *                  If 'pre', the sequence will be truncated from the beginning.
 *                  If 'post', the sequence will be truncated from the end.
 * @param inputKey The key for the sequence. Either 'tokens' or 'indexedTokens'.
 *                 The output key would be the same. Namely the original sequence will be
 *                 replaced by the shaped sequence.
 *                 Default is 'indexedTokens'.
 * @param padElement The element to be padded to the sequence if the original length is
 *                   smaller than the target length.
 *                   It should be a string if inputKey is 'tokens'.
 *                   It should be an integer if inputKey is 'indexedTokens'.
 *                   Default is 0 for 'indexedTokens' with the convention that we reserve index
 *                   0 for unknown words.
 */
class SequenceShaper(
    val len: Int,
    val truncMode: TruncMode = TruncMode.pre,
    val inputKey: String = TextFeature.indexedTokens,
    val padElement: Any = 0) extends TextTransformer {

  require(len > 0, "len should be positive")
  require(inputKey == TextFeature.indexedTokens || inputKey == TextFeature.tokens,
  "inputKey should be either tokens or indexedTokens")

  override def transform(feature: TextFeature): TextFeature = {
    require(feature.contains(inputKey), s"TextFeature doesn't contain $inputKey")
    val tokens = feature[Array[_]](inputKey)
    val shapedTokens = if (tokens.length > len) {
      truncMode match {
        case TruncMode.pre => tokens.slice(tokens.length - len, tokens.length)
        case TruncMode.post => tokens.slice(0, len)
        case _ => throw new IllegalArgumentException("Unknown truncation mode")
      }
    } else {
      if (inputKey == TextFeature.indexedTokens) {
        require(padElement.isInstanceOf[Int], "padElement should be an int for indexedTokens")
        tokens.asInstanceOf[Array[Int]] ++
          Array.fill[Int](len - tokens.length)(padElement.asInstanceOf[Int])
      }
      else {
        require(padElement.isInstanceOf[String], "padElement should be a string for tokens")
        tokens.asInstanceOf[Array[String]] ++
          Array.fill[String](len - tokens.length)(padElement.asInstanceOf[String])
      }
    }
    feature(inputKey) = shapedTokens
    feature
  }
}

object SequenceShaper {
  def apply(
      len: Int,
      truncMode: TruncMode = TruncMode.pre,
      inputKey: String = TextFeature.indexedTokens,
      padElement: Any = 0): SequenceShaper = {
    new SequenceShaper(len, truncMode, inputKey, padElement)
  }
}

object TruncMode extends Enumeration {
  type TruncMode = Value
  val pre, post = Value
}
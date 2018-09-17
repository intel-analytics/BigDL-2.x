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

import org.apache.log4j.Logger
import scala.collection.{Set, mutable}

/**
 * Each TextFeature keeps information of a single text record.
 * It can include various status of a text,
 * e.g. original text content, category label, words after tokenization,
 * index representation of tokens, BigDL Sample representation and so on.
 * It uses a HashMap to store all these data.
 * Each key is a string that can be used to identify the corresponding value.
 */
class TextFeature extends Serializable {
  import TextFeature.logger

  private def this(text: String, label: Option[Int]) {
    this
    require(text != null, "text for a TextFeature can't be null")
    state(TextFeature.text) = text
    if (label.nonEmpty) {
      state(TextFeature.label) = label.get
    }
  }

  private val state = new mutable.HashMap[String, Any]()

  def contains(key: String): Boolean = state.contains(key)

  def apply[T](key: String): T = {
    if (contains(key)) {
      state(key).asInstanceOf[T]
    } else {
      logger.warn(s"TextFeature doesn't have contain $key")
      null.asInstanceOf[T]
    }
  }

  def update(key: String, value: Any): Unit = state(key) = value

  def keys(): Set[String] = state.keySet

  /**
   * Whether the TextFeature contains label.
   */
  def hasLabel: Boolean = state.contains(TextFeature.label)

  /**
   * Get the label of the TextFeature.
   * If no label is stored, -1 will be returned.
   */
  def getLabel: Int = {
    if (hasLabel) {
      apply[Int](TextFeature.label)
    }
    else {
      logger.warn("No label is stored in the TextFeature")
      -1
    }
  }

  /**
   * Get the text content of the TextFeature.
   */
  def getText: String = apply[String](TextFeature.text)
}

object TextFeature {
  /**
   * Key for the original text content which should not be modified.
   * Value should be a String.
   */
  val text = "text"
  /**
   * Key for the label for the original text content.
   * Value should be an integer.
   */
  val label = "label"
  /**
   * Key for the tokens after doing tokenization (or other token-based transformation such as
   * normalization) on the original text.
   * Value should be an array of String.
   */
  val tokens = "tokens"
  /**
   * Key for the indices corresponding to the tokens after performing word2idx.
   * Value should be an array of integers.
   */
  val indexedTokens = "indexedTokens"
  /**
   * Key for the sample (feature and label if any).
   * Value should be a BigDL Sample.
   */
  val sample = "sample"
  /**
   * Key for the text prediction result.
   * Value should be a BigDL Activity.
   */
  val predict = "predict"

  val logger: Logger = Logger.getLogger(getClass)

  /**
   * Create a TextFeature without label.
   */
  def apply(text: String): TextFeature = {
    new TextFeature(text, None)
  }

  /**
   * Create a TextFeature with label.
   */
  def apply(text: String, label: Int): TextFeature = {
    new TextFeature(text, Some(label))
  }
}

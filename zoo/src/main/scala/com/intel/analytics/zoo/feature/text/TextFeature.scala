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

import scala.collection.{Set, mutable}

class TextFeature extends Serializable {
  def this(text: String, label: Option[Int] = None) {
    this
    state(TextFeature.text) = text
    if (label.nonEmpty) {
      state(TextFeature.label) = label
    }
  }

  private val state = new mutable.HashMap[String, Any]()

  def contains(key: String): Boolean = state.contains(key)

  def apply[T](key: String): T = {
    if (contains(key)) state(key).asInstanceOf[T] else null.asInstanceOf[T]
  }

  def update(key: String, value: Any): Unit = state(key) = value

  def keys(): Set[String] = state.keySet
}

object TextFeature {
  val text = "text" // String
  val label = "label" // Float
  val tokens = "tokens" // Array of String after tokenization
  val indexedTokens = "indexedTokens" // Array of int after word to index
  val tensor = "tensor" // vector representation if any
  val sample = "sample"
}

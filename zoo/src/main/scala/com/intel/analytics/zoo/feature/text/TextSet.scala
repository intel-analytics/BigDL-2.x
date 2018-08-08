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

import com.intel.analytics.zoo.feature.common.Preprocessing
import org.apache.spark.rdd.RDD

abstract class TextSet {

  def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    this.transform(transformer)
  }

  def isLocal(): Boolean

  def isDistributed(): Boolean

  def toLocal(): LocalTextSet = this.asInstanceOf[LocalTextSet]

  def toDistributed(): DistributedTextSet = this.asInstanceOf[DistributedTextSet]

}


object TextSet {

  def array(data: Array[TextFeature]): LocalTextSet = {
    new LocalTextSet(data)
  }

  def rdd(data: RDD[TextFeature]): DistributedTextSet = {
    new DistributedTextSet(data)
  }
}


class LocalTextSet(var array: Array[TextFeature]) extends TextSet {
  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    array = transformer.apply(array.toIterator).toArray
    this
  }

  override def isLocal(): Boolean = true

  override def isDistributed(): Boolean = false
}


class DistributedTextSet(var rdd: RDD[TextFeature]) extends TextSet {
  override def transform(transformer: Preprocessing[TextFeature, TextFeature]): TextSet = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

}

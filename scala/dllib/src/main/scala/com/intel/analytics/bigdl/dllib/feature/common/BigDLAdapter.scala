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
package com.intel.analytics.zoo.feature.common

import com.intel.analytics.bigdl.dataset.Transformer

/**
 * Convert a BigDL Transformer to a Preprocessing
 */
class BigDLAdapter[A, B] (bigDLTransformer: Transformer[A, B]) extends Preprocessing[A, B] {
  override def apply(prev: Iterator[A]): Iterator[B] = {
    bigDLTransformer(prev)
  }
}

object BigDLAdapter {
  def apply[A, B](bigDLTransformer: Transformer[A, B]): BigDLAdapter[A, B] =
    new BigDLAdapter(bigDLTransformer)
}

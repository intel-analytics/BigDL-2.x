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

package com.intel.analytics.zoo.models.textmatching

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.models.common.{Ranker, ZooModel}

import scala.reflect.ClassTag

/**
 * The base class for text matching models in Analytics Zoo.
 * Referred to MatchZoo implementation: https://github.com/NTMC-Community/MatchZoo
 */
abstract class TextMatcher[T](
    val text1Length: Int,
    val vocabSize: Int,
    val embedSize: Int = 300,
    val embedWeights: Tensor[T] = null,
    val trainEmbed: Boolean = true,
    val targetMode: String = "ranking")
  (implicit val tag: ClassTag[T], implicit val ev: TensorNumeric[T])
  extends ZooModel[Activity, Activity, T] with Ranker[T] {

  require(targetMode == "ranking" || targetMode == "classification", "targetMode should be " +
    s"either 'ranking' or 'classification', but got $targetMode")

}

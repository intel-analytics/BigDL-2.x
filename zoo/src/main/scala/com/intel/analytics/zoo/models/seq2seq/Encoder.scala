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

package com.intel.analytics.zoo.models.seq2seq

import com.intel.analytics.bigdl.nn.abstractnn.{Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * [[Encoder]] Base encoder class. Required by :obj:`com.intel.analytics.zoo.models.seq2seq`.
 */
abstract class Encoder[T: ClassTag](inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Activity, T](KerasUtils.addBatch(inputShape))
    with Net

/**
 * [[Decoder]] Base decoder class. Required by :obj:`com.intel.analytics.zoo.models.seq2seq`.
 */
abstract class Decoder[T: ClassTag](inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Tensor[T], T](KerasUtils.addBatch(inputShape))
    with Net

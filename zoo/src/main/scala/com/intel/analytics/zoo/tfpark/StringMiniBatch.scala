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
package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{Tensor, TensorNumericMath}
import org.apache.spark.rdd.RDD

class StringMiniBatch[T](data: Tensor[Array[Byte]]) extends MiniBatch[T] {
  override def size(): Int = data.nElement()

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    new StringMiniBatch[T](data.narrow(1, offset, length))
  }

  override def getInput(): Activity = {
    data
  }

  override def getTarget(): Activity = {
    null // fake target, should new be used
  }

  override def set(samples: Seq[Sample[T]])
                  (implicit ev: TensorNumericMath.TensorNumeric[T]): StringMiniBatch.this.type = {
    throw new UnsupportedOperationException("StringMiniBatch does not support set method")
  }
}

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

package com.intel.analytics.zoo.pipeline.api.keras.optimizers

import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor

trait SparseOptimMethod[@specialized(Float, Double) T] extends OptimMethod[T] {
  def optimize2(feval: (Array[Tensor[T]]) => (Array[T], Array[Tensor[T]]), parameter: Array[Tensor[T]])
  : (Array[Tensor[T]], Array[Array[T]]) = {
    throw new UnsupportedOperationException("Please use" +
      "optimize(feval: (Array[Tensor[T]]) => (Array[T], Array[Tensor[T]]), parameter: Array[Tensor[T]])" +
      "instead")
  }
}

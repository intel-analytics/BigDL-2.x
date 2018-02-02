/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.example.keras

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.keras._
import com.intel.analytics.bigdl.utils.Shape

object LeNet {
  def apply(): Module[Float] = {
    val model = Sequential[Float]()
    model.add(Reshape(Array(1, 28, 28), inputShape = Shape(28, 28, 1)))
    model.add(Convolution2D(32, 3, 3, activation = "relu"))
    model.add(Convolution2D(32, 3, 3, activation = "relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    model
  }
}

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

package com.intel.analytics.bigdl.example.tensorflow

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor

/**
 * This example show how to load a tensorflow model defined in slim and
 * use it to do prediction
 */
object Load {
  def main(args: Array[String]): Unit = {
    require(args.length == 1, "Please input the model path as the first argument")
    val model = Module.loadTF(args(0), Seq("Placeholder"), Seq("LeNet/fc4/BiasAdd"))
    val result = model.forward(Tensor(1, 1, 28, 28).rand())
    println(result)
  }
}

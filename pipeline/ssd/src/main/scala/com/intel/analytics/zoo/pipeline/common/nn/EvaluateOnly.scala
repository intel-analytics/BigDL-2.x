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

package com.intel.analytics.zoo.pipeline.common.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * If is a container to contral different behavior in training mode and evaluate mode
 * in training mode, the submodules will be skipped and input directly be passed to output;
 * in evaluation mode, the modules will process the input tensors.
 * @return this
 */
@SerialVersionUID(5600616321943671046L)
class EvaluateOnly[T: ClassTag](module: Module[T])(implicit ev: TensorNumeric[T])
  extends Container[Activity, Activity, T] {

  add(module)

  override def updateOutput(input: Activity): Activity = {
    output = if (!isTraining()) {
      module.updateOutput(input)
    } else {
      input
    }
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (isTraining()) {
      gradInput = gradOutput
    }
    gradInput
  }

  override def toString: String = "nn.EvaluateOnly"
}

object EvaluateOnly {
  def apply[@specialized(Float, Double) T: ClassTag]
  (module: Module[T])(implicit ev: TensorNumeric[T]): EvaluateOnly[T] = new EvaluateOnly[T](module)
}

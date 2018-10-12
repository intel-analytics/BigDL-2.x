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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[zoo] class InternalRecurrent[T: ClassTag](
    batchNormParams: BatchNormParams[T] = null,
    maskZero: Boolean = false
)(implicit ev: TensorNumeric[T]) extends com.intel.analytics.bigdl.nn.Recurrent[T](batchNormParams, maskZero) {

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    super.add(module)
    if (this.preTopology != null) {
      module.asInstanceOf[Cell[T]].preTopology = null
    }
    this
  }

  // get gradient hidden state at the first time step
  def getGradHiddenState(): Activity = {
    require(cells != null && cells(0).gradInput != null,
      "getGradHiddenState need to be called after backward")
    cells(0).gradInput.toTable(hidDim)
  }

  protected var initGradHiddenState: Activity = null
  // set gradient hiddent state at the last time step
  def setGradHiddenState(gradHiddenState: Activity): Unit = {
    initGradHiddenState = gradHiddenState
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    if (initGradHiddenState != null) gradHidden = initGradHiddenState
    super.updateGradInput(input, gradOutput)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (initGradHiddenState != null) gradHidden = initGradHiddenState
    super.updateGradInput(input, gradOutput)
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val st = System.nanoTime
    gradInput = updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    this.backwardTime = System.nanoTime - st
    gradInput
  }
}

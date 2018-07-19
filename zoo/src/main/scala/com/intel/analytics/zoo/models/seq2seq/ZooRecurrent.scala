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

import com.intel.analytics.bigdl.nn.{BatchNormParams, Recurrent, RecurrentDecoder, Utils}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.nn.BigDLWrapper

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * [[ZooRecurrent]] module is a container of rnn cells
 * Different types of rnn cells can be added using add() function
 */
class ZooRecurrent[T : ClassTag](batchNormParams: BatchNormParams[T] = null,
                                 maskZero: Boolean = false)(implicit ev: TensorNumeric[T])
  extends Recurrent[T](batchNormParams, maskZero) {
  protected var maskBuffer: Tensor[T] = Tensor()
  protected var gradOutputBuff: Table = T()
//  protected var indexBuffer: Tensor[T] = Tensor()
//  protected var inputBuffer: Tensor[T] = Tensor()
//  protected var outputBuffers: ArrayBuffer[Tensor[T]] = ArrayBuffer(Tensor())
  protected var minLength: Int = 0

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

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val st = System.nanoTime
    var i = times

    while (i >= 1) {
      currentGradOutput(hidDim) = if (i != times) cells(i).gradInput.toTable(hidDim)
      else if (initGradHiddenState == null) gradHidden else initGradHiddenState
      currentGradOutput(inputDim) = BigDLWrapper.copy(gradOutput, i, stepGradBuffer)

      _input(hidDim) = if (i > 1) cells(i - 2).output.toTable(hidDim)
      else if (initHiddenState == null) hidden else initHiddenState

      _input(inputDim) = BigDLWrapper.copy(input2Cell, i, stepInput2CellBuf)

      if (i == 1) {
        cells(i - 1).regluarized(true)
      } else {
        cells(i - 1).regluarized(false)
      }

      if (maskZero && i > minLength) {
        val curMask = maskBuffer.select(2, i)
        if (gradOutputBuff.length() == 0) {
          Utils.recursiveResizeAs(gradOutputBuff, currentGradOutput)
        }
        Utils.recursiveCopy(gradOutputBuff, currentGradOutput)
        for (b <- 1 to curMask.size(1)) {
          if (curMask(Array(b, 1)) == ev.zero) {
            val originState = gradOutputBuff[Table](Recurrent.hidDim)
            for (j <- 1 to originState.length()) {
              originState[Tensor[T]](j).select(1, b).zero()
            }
          }
        }

        cells(i - 1).backward(_input, gradOutputBuff).toTable

        for (b <- 1 to curMask.size(1)) {
          if (curMask(Array(b, 1)) == ev.zero) {
            val newState = cells(i - 1).gradInput[Table](hidDim)
            val originState = currentGradOutput[Table](hidDim)
            for (j <- 1 to newState.length()) {
              newState[Tensor[T]](j).select(1, b).copy(originState[Tensor[T]](j).select(1, b))
            }
          }
        }
      } else {
        cells(i - 1).backward(_input, currentGradOutput)
      }
      cells(i - 1).backward(_input, currentGradOutput)
      i -= 1
    }

    gradInput = if (preTopology != null) {
      /**
        * if preTopology is Sequential, it has not created gradInput.
        * Thus, it needs to create a new Tensor.
        */
      if (preTopology.gradInput == null) {
        preTopology.gradInput = Tensor[T]()
      }
      preTopology.gradInput.toTensor[T]
    } else {
      gradInput2Cell
    }
    gradInput2Cell.resizeAs(input2Cell)
    BigDLWrapper.copy(cells.map(x => x.gradInput.toTable[Tensor[T]](inputDim)), gradInput2Cell)

    if (preTopology != null) {
      gradInput = preTopology.backward(input, gradInput2Cell).toTensor[T]
    }

    this.backwardTime = System.nanoTime - st
    gradInput
  }
}


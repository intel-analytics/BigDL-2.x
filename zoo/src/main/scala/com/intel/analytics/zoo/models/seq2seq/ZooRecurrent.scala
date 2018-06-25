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

import com.intel.analytics.bigdl.nn.{Recurrent, RecurrentDecoder}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.wrapper.BigDLWrapper

import scala.reflect.ClassTag

/**
 * [[ZooRecurrent]] module is a container of rnn cells
 * Different types of rnn cells can be added using add() function
 */
class ZooRecurrent[T : ClassTag]()(implicit ev: TensorNumeric[T]) extends Recurrent[T] {
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

/**
 * [[ZooRecurrentDecoder]] module is a container of rnn cells that used to make
 * a prediction of the next timestep based on the prediction we made from
 * the previous timestep. Input for RecurrentDecoder is dynamically composed
 * during training. input at t(i) is based on output at t(i-1), input at t(0) is
 * user input, and user input has to be batch x stepShape(shape of the input
 * at a single time step).

 * Different types of rnn cells can be added using add() function.
 * @param seqLen max sequence length of the output
 * @param stopSign if prediction is the same with stopSign, it will stop predict
 * @param loopFunc before feeding output at last time step to next time step,
 *                 pass it through loopFunc
 */
class ZooRecurrentDecoder[T : ClassTag](seqLen: Int, stopSign: Tensor[T] = null,
  loopFunc: (Tensor[T]) => (Tensor[T]) = null)(implicit ev: TensorNumeric[T])
  extends RecurrentDecoder[T](seqLen) {
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

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim == 2 || input.dim == 4 || input.dim == 5,
      "Recurrent: input should be a 2D/4D/5D Tensor, e.g [batch, nDim], " +
        s"current input.dim = ${input.dim}")
    batchSize = input.size(batchDim)
    val hiddenSize = topology.hiddensShape(0)
    val outputSize = input.size()
    require(hiddenSize == input.size()(1), "hiddenSize is " +
      "not the same with input size!! Please update cell settings or use Recurrent instead!")
    val featureSizes = outputSize.drop(1)
    output.resize(Array(batchSize, times) ++ featureSizes)
    // Clone N modules along the sequence dimension.
    initHidden(featureSizes)
    cloneCells()

    var i = 1
    // input at t(0) is user input
    currentInput = if (initHiddenState != null) T(input, initHiddenState)
    else T(input, hidden)
    cells(i - 1).updateOutput(currentInput)
    BigDLWrapper.copy(cells(i - 1).output.toTable[Tensor[T]](inputDim), output, i)
    i += 1
    while (i <= times && (stopSign == null ||
      !cells(i - 2).output.toTable[Tensor[T]](inputDim).almostEqual(stopSign, 1e-6))) {
      // input at t(i) is output at t(i-1)
      if (loopFunc != null) {
        val preOutput = cells(i - 2).output.toTable[Tensor[T]](inputDim)
        preOutput.copy(loopFunc(preOutput))
      }
      currentInput = cells(i - 2).output
      cells(i - 1).updateOutput(currentInput)
      BigDLWrapper.copy(cells(i - 1).output.toTable[Tensor[T]](inputDim), output, i)
      i += 1
    }
    times = i - 1
    output = output.narrow(timeDim, 1, times)
    output
  }

  override def backward(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val st = System.nanoTime
    gradInput.resizeAs(output)
    currentGradOutput(hidDim) = if (initGradHiddenState == null) gradHidden else initGradHiddenState
    var i = times
    while (i >= 1) {
      currentGradOutput(inputDim) = if (i == times) {
        BigDLWrapper.copy(gradOutput, i, stepGradBuffer)
      } else {
        val _gradInput = cells(i).gradInput.toTable[Tensor[T]](inputDim)
        BigDLWrapper.copy(gradOutput, i, stepGradBuffer).add(_gradInput)
      }

      _input = if (i == 1) {
        if (initHiddenState == null) T(input, hidden)
        else T(input, initHiddenState)
      } else cells(i - 2).output

      if (i == 1) {
        cells(i - 1).regluarized(true)
      } else {
        cells(i - 1).regluarized(false)
      }
      cells(i - 1).backward(_input, currentGradOutput)
      currentGradOutput(hidDim) = cells(i - 1).gradInput.toTable(hidDim)
      i -= 1
    }
    BigDLWrapper.copy(cells.map(x => x.gradInput.toTable[Tensor[T]](inputDim)), gradInput)
    this.backwardTime = System.nanoTime - st
    gradInput
  }
}

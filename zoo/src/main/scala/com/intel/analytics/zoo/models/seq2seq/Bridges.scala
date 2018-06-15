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

import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.nn.{Recurrent, RecurrentDecoder}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.reflect.ClassTag

abstract class Bridge {
  def forwardStates[T: ClassTag](encoder: Array[ZooRecurrent[T]],
    decoder: Array[Recurrent[T]])(implicit ev: TensorNumeric[T]): Unit

  def backwardStates[T: ClassTag](encoder: Array[ZooRecurrent[T]],
    decoder: Array[Recurrent[T]])(implicit ev: TensorNumeric[T]): Unit
}

class ZeroBridge() extends Bridge {
  override def forwardStates[T: ClassTag](encoder: Array[ZooRecurrent[T]],
    decoder: Array[Recurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {}

  override def backwardStates[T: ClassTag](encoder: Array[ZooRecurrent[T]],
    decoder: Array[Recurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {}
}

class PassThroughBridge() extends Bridge {
  override def forwardStates[T: ClassTag](encoder: Array[ZooRecurrent[T]],
    decoder: Array[Recurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {
    if (decoder.head.isInstanceOf[ZooRecurrentDecoder[T]]) {
      if (encoder.size == 1) {
        decoder.head.setHiddenState(encoder.head.getHiddenState())
      } else {
        val hiddenState = T()
        for((rec, i) <- encoder.view.zipWithIndex) {
          hiddenState(i + 1) = rec.getHiddenState()
        }
        decoder.head.setHiddenState(hiddenState)
      }
    } else {
      for ((x, i) <- encoder.view.zipWithIndex) {
        decoder(i).setHiddenState(x.getHiddenState())
      }
    }
  }

  override def backwardStates[T: ClassTag](encoder: Array[ZooRecurrent[T]],
    decoder: Array[Recurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {
    if (decoder.head.isInstanceOf[ZooRecurrentDecoder[T]]) {
      val gradHiddenStates = decoder.head.asInstanceOf[ZooRecurrentDecoder[T]].getGradHiddenState()
      if (encoder.size == 1) {
        encoder.head.setGradHiddenState(gradHiddenStates)
      } else {
        for ((x, i) <- encoder.view.zipWithIndex) {
          x.setGradHiddenState(gradHiddenStates.toTable(i + 1))
        }
      }
    } else {
      for ((x, i) <- decoder.view.zipWithIndex) {
        encoder(i).setGradHiddenState(x.asInstanceOf[ZooRecurrent[T]].getGradHiddenState())
      }
    }
  }
}

class InitialStateBridge[T: ClassTag](val activations: Array[Array[TensorModule[T]]])
                                     (implicit ev: TensorNumeric[T]) extends Bridge {
  override def forwardStates[T: ClassTag](encoder: Array[ZooRecurrent[T]],
    decoder: Array[Recurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {
    if (decoder.head.isInstanceOf[ZooRecurrentDecoder[T]]) {
      if (encoder.size == 1) {
        require(activations.head != null, "Please use PassThroughBridge instead")
        decoder.head.setHiddenState(updateState(encoder.head.getHiddenState(), activations.head))
      } else {
        val hiddenState = T()
        for((rec, i) <- encoder.view.zipWithIndex) {
          hiddenState(i + 1) = if (activations(i) != null)
            updateState(rec.getHiddenState(), activations(i)) else rec.getHiddenState()
        }
        decoder.head.setHiddenState(hiddenState)
      }
    } else {
      for ((x, i) <- encoder.view.zipWithIndex) {
        val newState = if (activations(i) != null) updateState(x.getHiddenState(), activations(i))
        else x.getHiddenState()
        decoder(i).setHiddenState(newState)
      }
    }
  }

  override def backwardStates[T: ClassTag](encoder: Array[ZooRecurrent[T]],
    decoder: Array[Recurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {
    if (decoder.head.isInstanceOf[ZooRecurrentDecoder[T]]) {
      val gradHiddenStates = decoder.head.asInstanceOf[ZooRecurrentDecoder[T]].getGradHiddenState()
      if (encoder.size == 1) {
        encoder.head.setGradHiddenState(updateGradState(encoder.head.getHiddenState(),
          gradHiddenStates, activations.head))
      } else {
        for ((x, i) <- encoder.view.zipWithIndex) {
          val newGradHiddenState = if (activations(i) != null) updateGradState(x.getHiddenState(),
              gradHiddenStates.toTable(i + 1), activations(i)) else gradHiddenStates.toTable(i + 1)
            x.setGradHiddenState(newGradHiddenState)
        }
      }
    } else {
      for ((x, i) <- decoder.view.zipWithIndex) {
        val gradHiddenState = x.asInstanceOf[ZooRecurrent[T]].getGradHiddenState()
        val newGradHiddenState = if (activations(i) != null) updateGradState(encoder(i).getHiddenState(),
          gradHiddenState, activations(i)) else gradHiddenState
        encoder(i).setGradHiddenState(newGradHiddenState)
      }
    }
  }

  private def updateState(state: Activity,
    activation: Array[TensorModule[T]]): Activity = {
    require(activation != null, "activation cannot be null")
    var newState: Activity = null
    if (state.isTensor) {
      require(activation.head != null, "activation cannot be null")
      newState = activation.head.forward(state.toTensor[T])
    } else {
      require(activation(0) != null && activation(1) != null, "activation cannot be null")
      newState = T()
      newState.toTable(1) = activation(0).forward(state.toTable(1))
      newState.toTable(2) = activation(1).forward(state.toTable(2))
    }
    newState
  }

  private def updateGradState(state: Activity, gradState: Activity,
    activation: Array[TensorModule[T]]): Activity = {
    require(activation != null, "activation cannot be null")
    var newGradState: Activity = null
    if (gradState.isTensor) {
      require(activation.head != null, "activation cannot be null")
      newGradState = activation.head.backward(state.toTensor[T], gradState.toTensor[T])
    } else {
      require(activation(0) != null && activation(1) != null, "activations cannot be null")
      newGradState = T()
      newGradState.toTable(1) = activation(0).backward(state.toTable(1), gradState.toTable(1))
      newGradState.toTable(2) = activation(1).backward(state.toTable(2), gradState.toTable(2))
    }
    newGradState
  }
}

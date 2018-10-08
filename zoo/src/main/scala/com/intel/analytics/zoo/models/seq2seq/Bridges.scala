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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.nn.{Recurrent, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.pipeline.api.keras.layers.internal.InternalRecurrent

import scala.reflect.ClassTag

/**
 * [[Bridge]] defines how to pass state between encoder and decoder
 */
abstract class Bridge extends Serializable {
  /**
   * pass encoder state to decoder in forward
   * @param encoder array of Recurrent or its subclasses used in encoder
   * @param decoder array of Recurrent or its subclasses used in decoder
   * @return
   */
  def forwardStates[T: ClassTag](encoder: Array[InternalRecurrent[T]],
    decoder: Array[InternalRecurrent[T]])(implicit ev: TensorNumeric[T]): Unit

  /**
   * pass decoder state to encoder in backward
   * @param encoder array of Recurrent or its subclasses used in encoder
   * @param decoder array of Recurrent or its subclasses used in decoder
   * @return
   */
  def backwardStates[T: ClassTag](encoder: Array[InternalRecurrent[T]],
    decoder: Array[InternalRecurrent[T]])(implicit ev: TensorNumeric[T]): Unit

  /**
   * Modules used in the Bridge, usually for get parameters in the Bridge
   * @return Modules used in the Bridge
   */
  def toModel[T: ClassTag](implicit ev: TensorNumeric[T]): Sequential[T] = Sequential[T]()
}

/**
 * [[ZeroBridge]] doesn't pass state between encoder and decoder. The init decoder state is 0
 */
class ZeroBridge() extends Bridge {
  override def forwardStates[T: ClassTag](encoder: Array[InternalRecurrent[T]],
    decoder: Array[InternalRecurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {}

  override def backwardStates[T: ClassTag](encoder: Array[InternalRecurrent[T]],
    decoder: Array[InternalRecurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {}
}

/**
 * [[PassThroughBridge]] pass state between encoder and decoder without change.
 * Requires encoder states are the same size with decoder states
 */
class PassThroughBridge() extends Bridge {
  override def forwardStates[T: ClassTag](encoder: Array[InternalRecurrent[T]],
    decoder: Array[InternalRecurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {
    require(encoder.size == decoder.size, "encoder should be the same size with decoder")
    for ((x, i) <- encoder.view.zipWithIndex) {
      decoder(i).setHiddenState(x.getHiddenState())
    }
  }

  override def backwardStates[T: ClassTag](encoder: Array[InternalRecurrent[T]],
    decoder: Array[InternalRecurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {
    for ((x, i) <- decoder.view.zipWithIndex) {
      encoder(i).setGradHiddenState(x.asInstanceOf[InternalRecurrent[T]].getGradHiddenState())
    }
  }
}

/**
 * [[InitialStateBridge]] Init decoder state with passing encoder state through
 * activations. It allows encoder states are in different size with decoder states
 */
class InitialStateBridge[T: ClassTag](val activations: Array[Array[TensorModule[T]]])
                                     (implicit ev: TensorNumeric[T]) extends Bridge {
  override def toModel[T: ClassTag](implicit ev: TensorNumeric[T]): Sequential[T] = {
    val model = Sequential[T]()
    activations.filter(_ != null).flatten.foreach(x =>
      model.add(x.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]))
    model
  }

  override def forwardStates[T: ClassTag](encoder: Array[InternalRecurrent[T]],
    decoder: Array[InternalRecurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {
    for ((x, i) <- encoder.view.zipWithIndex) {
      val newState = if (activations(i) != null) updateState(x.getHiddenState(), activations(i))
      else x.getHiddenState()
      decoder(i).setHiddenState(newState)
    }
  }

  override def backwardStates[T: ClassTag](encoder: Array[InternalRecurrent[T]],
    decoder: Array[InternalRecurrent[T]])(implicit ev: TensorNumeric[T]): Unit = {
    for ((x, i) <- decoder.view.zipWithIndex) {
      val gradHiddenState = x.getGradHiddenState()
      val newGradHiddenState = if (activations(i) != null) {
        updateGradState(encoder(i).getHiddenState(), gradHiddenState, activations(i))
      } else gradHiddenState
      encoder(i).setGradHiddenState(newGradHiddenState)
    }
  }

  private def updateState(state: Activity,
    activation: Array[TensorModule[T]]): Activity = {
    require(activation != null, "activation cannot be null")
    require(activation.size == 1 || activation.size == 2, "state size of rnn must be 1|2")
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

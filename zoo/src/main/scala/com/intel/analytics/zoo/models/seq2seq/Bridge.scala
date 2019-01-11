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

import com.intel.analytics.bigdl.nn.abstractnn.{Activity, AbstractModule}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._

import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential

import scala.reflect.ClassTag

/**
 * [[Bridge]] private class which defines how to transform encoder to decoder
 * @param bridgeType currently only support "dense | densenonlinear | customized"
 * @param decoderHiddenSize hidden size of decoder
 * @param bridge keras layers used to do the transformation
 */
class Bridge[T: ClassTag] private[zoo] (val bridgeType: String,
  var decoderHiddenSize: Int,
  bridge: KerasLayer[Tensor[T], Tensor[T], T])(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T]() with Net {

  def this(bridge: KerasLayer[Tensor[T], Tensor[T], T])(implicit ev: TensorNumeric[T]) =
    this("customized", 0, bridge)

  def this(bridgeType: String, decoderHiddenSize: Int)(implicit ev: TensorNumeric[T]) =
    this(bridgeType, decoderHiddenSize, null)

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = {
    val layerNum = inputShape.toMulti().size
    val stateNum = inputShape.toMulti().head.toMulti().size

    val _inputShape = KerasUtils.removeBatch(inputShape)

    val layer = Sequential()
    if (stateNum > 1 || layerNum > 1) {
      val flattenShape = _inputShape.toMulti().map(_.toMulti().map(_.toSingle())).flatten.flatten

      val inputLayers = flattenShape.map(x => InputLayer(Shape(Array(x))))
      layer.add(Merge(inputLayers, mode = "concat"))
    } else layer.add(InputLayer(_inputShape))

    // construct bridge
    val _bridge = bridgeType.toLowerCase() match {
      case "dense" =>
        Dense(decoderHiddenSize * stateNum * layerNum, bias = false)
      case "densenonlinear" =>
        Dense(decoderHiddenSize * stateNum * layerNum, activation = "tanh", bias = false)
      case "customized" =>
        bridge
      case _ => throw new IllegalArgumentException(s"Only support dense | densenonlinear" +
        s" as bridgeType. For customized bridge, please use " +
        s"Bridge(bridge: KerasLayer[Tensor[T], Tensor[T], T]) to create a bridge")
    }

    layer.add(_bridge)

    if (layerNum > 1 || stateNum > 1) {
      layer.add(SplitTensor[T](Bridge.splitDim, layerNum * stateNum))
    }

    layer.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }

  private def updateShape(inputShape: Shape): Shape = {
    if (inputShape.isInstanceOf[SingleShape]) {
      val sizes = inputShape.toSingle()
      Shape(Array(sizes(0), decoderHiddenSize) ++ sizes.drop(2))
    } else {
      MultiShape(inputShape.toMulti().map(updateShape(_)))
    }
  }

  private def constructTensor(inputShape: Shape): Activity = {
    if (inputShape.isInstanceOf[SingleShape]) {
      val singleShape = inputShape.toSingle().toArray
      Tensor(Array(1) ++ singleShape.drop(1)).fill(ev.one)
    }
    else {
      T.array(inputShape.toMulti().map(constructTensor(_)).toArray)
    }
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    if (decoderHiddenSize == 0) {
      val _input = constructTensor(inputShape)
      val _output = updateOutput(_input)
      decoderHiddenSize = _output.toTable.get[Table](1).get.get[Tensor[T]](1).get.size(2)
    }
    MultiShape(inputShape.toMulti().map(updateShape(_)))
  }

  override def updateOutput(input: Activity): Activity = {
    val _input = input.toTable.flatten()
    val _output = labor.forward(_input)

    output = _output.toTable.inverseFlatten(input.toTable)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    val (_input, _gradOutput) = (input.toTable.flatten(), gradOutput.toTable.flatten())
    val _gradInput = labor.backward(_input, _gradOutput)
    gradInput = if (_input.length() > 1) {
      val reshapedGradInput = if (input.toTable.length() == 1) T(_gradInput)
      else _gradInput.toTable.inverseFlatten(input.toTable)
      T(reshapedGradInput, T())
    } else _gradInput.toTable
    gradInput
  }
}

object Bridge {
  /**
   * [[Bridge]] defines how to transform encoder to decoder
   * @param bridgeType currently only support "dense | densenonlinear"
   * @param decoderHiddenSize hidden size of decoder
   */
  def apply[@specialized(Float, Double) T: ClassTag](bridgeType: String,
    decoderHiddenSize: Int)(implicit ev: TensorNumeric[T]):
  KerasLayer[Activity, Activity, T] = {
    require(decoderHiddenSize > 0, "invalid decoderHiddenSize")
    new Bridge(bridgeType, decoderHiddenSize)
  }

  /**
   * [[Bridge]] defines how to transform encoder to decoder
   * @param bridge keras layers used to do the transformation
   */
  def apply[@specialized(Float, Double) T: ClassTag](bridge: KerasLayer[Tensor[T], Tensor[T], T])
    (implicit ev: TensorNumeric[T]): KerasLayer[Activity, Activity, T] = {
    new Bridge(bridge)
  }

  val splitDim = 1
}

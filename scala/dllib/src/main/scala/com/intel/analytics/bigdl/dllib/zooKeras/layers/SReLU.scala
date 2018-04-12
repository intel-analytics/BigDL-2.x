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

import com.intel.analytics.bigdl.nn.{InitializationMethod, Ones, Xavier, Zeros}
import com.intel.analytics.bigdl.nn.keras.{SReLU => BigDLSReLU}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

/**
 * S-shaped Rectified Linear Unit.
 * It follows:
 * f(x) = t^r + a^r(x - t^r) for x >= t^r,
 * f(x) = x for t^r > x > t^l,
 * f(x) = t^l + a^l(x - t^l) for x <= t^l.
 *
 * When you use this layer as the first layer of a model, you need to provide the argument
 * inputShape (a Single Shape, does not include the batch dimension).
 *
 * @param tLeftInit Initialization function for the left part intercept. Default is Zeros.
 *                  You can also pass in corresponding string representations such as 'zero'
 *                  or 'normal', etc. for simple init methods in the factory method.
 * @param aLeftInit Initialization function for the left part slope. Default is Xavier.
 *                  You can also pass in corresponding string representations such as
 *                  'glorot_uniform', etc. for simple init methods in the factory method.
 * @param tRightInit Initialization function for the right part intercept. Default is Xavier.
 *                   You can also pass in corresponding string representations such as
 *                  'glorot_uniform', etc. for simple init methods in the factory method.
 * @param aRightInit Initialization function for the right part slope. Default is Ones.
 *                   You can also pass in corresponding string representations such as 'one'
 *                   or 'normal', etc. for simple init methods in the factory method.
 * @param sharedAxes Array of Int. The axes along which to share learnable parameters
 *                   for the activation function. Default is null.
 *                   For example, if the incoming feature maps are from a 2D convolution
 *                   with output shape (batch, height, width, channels),
 *                   and you wish to share parameters across space
 *                   so that each filter only has one set of parameters,
 *                   set 'sharedAxes = Array(1,2)'.
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now.
 */
class SReLU[T: ClassTag](
    override val tLeftInit: InitializationMethod = Zeros,
    override val aLeftInit: InitializationMethod = Xavier,
    override val tRightInit: InitializationMethod = Xavier,
    override val aRightInit: InitializationMethod = Ones,
    override val sharedAxes: Array[Int] = null,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BigDLSReLU[T](
    tLeftInit, aLeftInit, tRightInit, aRightInit, sharedAxes, inputShape) {
}

object SReLU {
  def apply[@specialized(Float, Double) T: ClassTag](
    tLeftInit: String = "zero",
    aLeftInit: String = "glorot_uniform",
    tRightInit: String = "glorot_uniform",
    aRightInit: String = "one",
    sharedAxes: Array[Int] = null,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SReLU[T] = {
    new SReLU[T](KerasUtils.getInitMethod(tLeftInit), KerasUtils.getInitMethod(aLeftInit),
      KerasUtils.getInitMethod(tRightInit), KerasUtils.getInitMethod(aRightInit),
      sharedAxes, inputShape)
  }
}

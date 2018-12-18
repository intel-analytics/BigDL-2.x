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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.optim.SGD.{Default, LearningRateSchedule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.math._
import scala.reflect.ClassTag

/**
 * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf with learning rate schedule.
 * @param learningRate learning rate
 * @param learningRateDecay learning rate decay
 * @param learningRateSchedule learning rate schedule
 * @param beta1 first moment coefficient
 * @param beta2 second moment coefficient
 * @param Epsilon for numerical stability
 */
class AdamWithSchedule[@specialized(Float, Double) T: ClassTag](
  learningRate: Double = 1e-3,
  learningRateDecay: Double = 0.0,
  learningRateSchedule: LearningRateSchedule = Default(),
  var beta1: Double = 0.9,
  var beta2: Double = 0.999,
  var Epsilon: Double = 1e-8)(implicit ev: TensorNumeric[T])
  extends SGD[T](learningRate = learningRate,
    learningRateDecay = learningRateDecay, learningRateSchedule = learningRateSchedule) {

  @transient
  private var buffer: Tensor[T] = null

  /**
   * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
               parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    this.updateHyperParameter()
    if (buffer == null) buffer = Tensor[T]()
    val lr = this.learningRate
    val lrd = this.learningRateDecay
    val beta1 = this.beta1
    val beta2 = this.beta2
    val eps = this.Epsilon

    val (fx, dfdx) = feval(parameter)

    var timestep = state.getOrElse[Int]("evalCounter", 0)
    val (_s, _r, _denom) =
      if (state.get[Tensor[T]]("s").isDefined) {
        (state.get[Tensor[T]]("s").get, state.get[Tensor[T]]("r").get,
          state.get[Tensor[T]]("denom").get.resizeAs(dfdx))
      } else {
        (Tensor[T]().resizeAs(dfdx).zero(), Tensor[T]().resizeAs(dfdx).zero(),
          Tensor[T]().resizeAs(dfdx).zero())
      }

    val clr = - this.learningRateSchedule.currentRate
//    val clr = lr / (1 + timestep*lrd)
    timestep = timestep + 1

    /**
     * m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
     * v_t = beta_2 * v_t-1 + (1 - beta_2) * g_t * g_t
     */
    _s.mul(ev.fromType[Double](beta1)).add(ev.fromType[Double](1-beta1), dfdx)
    // buffer = dfdx * dfdx
    buffer.resizeAs(dfdx).cmul(dfdx, dfdx)
    _r.mul(ev.fromType[Double](beta2)).add(ev.fromType[Double](1-beta2), buffer)
    _denom.sqrt(_r)

    // used as MKL.axpy: 1 * a + y = y, and fill buffer with one
    buffer.fill(ev.one)
    _denom.add(ev.fromType(eps), buffer)

    // efficiency improved upon by changing the order of computation, at expense of clarity
    val biasCorrection1 = 1 - pow(beta1, timestep)
    val biasCorrection2 = 1 - pow(beta2, timestep)
    val stepSize = clr * sqrt(biasCorrection2) / biasCorrection1
    parameter.addcdiv(ev.fromType[Double](-stepSize), _s, _denom)

    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon
    state("s") = _s // 1st moment variables
    state("r") = _r // 2nd moment variables
    state("denom") = _denom // 3nd moment variables

    (parameter, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    super.loadFromTable(config)
    this.beta1 = config.get[Double]("beta1").getOrElse(this.beta1)
    this.beta2 = config.get[Double]("beta2").getOrElse(this.beta2)
    this.Epsilon = config.get[Double]("Epsilon").getOrElse(this.Epsilon)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("s")
    state.delete("r")
  }
}

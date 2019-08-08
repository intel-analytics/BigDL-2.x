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
package com.intel.analytics.zoo.pipeline.api.keras.optimizers

import com.intel.analytics.bigdl.optim.SGD.{Default, LearningRateSchedule}
import com.intel.analytics.bigdl.tensor.{IndexedSlicesTensor, SparseTensorUtils, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.SGDRef

import scala.math._
import scala.reflect.ClassTag


/**
 * An implementation of Adam http://arxiv.org/pdf/1412.6980.pdf with learning rate schedule.
 * @param lr learning rate
 * @param decay learning rate decay
 * @param schedule learning rate schedule
 * @param beta_1 first moment coefficient
 * @param beta_2 second moment coefficient
 * @param epsilon for numerical stability
 */
class IndexedSlicesAdam[@specialized(Float, Double) T: ClassTag](
  lr: Double = 1e-3,
  beta_1: Double = 0.9,
  beta_2: Double = 0.999,
  epsilon: Double = 1e-8,
  decay: Double = 0.0,
  schedule: LearningRateSchedule = Default()
)(implicit ev: TensorNumeric[T]) extends Adam[T](lr, beta_1, beta_2, epsilon, decay, schedule)
  with SparseOptimMethod[T] {

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
  override def optimize2(feval: (Array[Tensor[T]]) => (Array[T], Array[Tensor[T]]), parameter: Array[Tensor[T]])
  : (Array[Tensor[T]], Array[Array[T]]) = {
    this.updateHyperParameter()
    val lr = this.lr
    val lrd = this.decay
    val beta1 = this.beta_1
    val beta2 = this.beta_2
    val eps = this.epsilon

    val (fx, dfdx) = feval(parameter)
    val state = SGDRef.getstate(this)
    var timestep = state.getOrElse[Int]("neval", 0)

    var i = 0
    while (i < parameter.length) {
      val (_s, _r, _denom) =
        if (state.get[Tensor[T]]("s").isDefined) {
          (state.get[Tensor[T]](s"s${i}").get, state.get[Tensor[T]](s"r${i}").get,
            state.get[Tensor[T]](s"denom${i}").get.resizeAs(dfdx(i)))
        } else {
          (IndexedSlicesTensor[T](), IndexedSlicesTensor[T](), IndexedSlicesTensor[T]())
        }

      val clr = - this.schedule.currentRate

      /**
       * m_t = beta_1 * m_t-1 + (1 - beta_1) * g_t
       * v_t = beta_2 * v_t-1 + (1 - beta_2) * g_t * g_t
       */
      _s.mul(ev.fromType[Double](beta1))
      _s.add(ev.fromType[Double](1-beta1), dfdx(i))
      // buffer = dfdx * dfdx
      buffer = dfdx(i).clone().cmul(dfdx(i))
      _r.mul(ev.fromType[Double](beta2)).add(ev.fromType[Double](1-beta2), buffer)
      _denom.sqrt(_r)

      _denom.add(ev.fromType(eps))

      // efficiency improved upon by changing the order of computation, at expense of clarity
      val biasCorrection1 = 1 - pow(beta1, timestep)
      val biasCorrection2 = 1 - pow(beta2, timestep)
      val stepSize = clr * sqrt(biasCorrection2) / biasCorrection1
      SparseTensorUtils.addcdivSparseTensor(parameter(i), ev.fromType[Double](-stepSize), _s, _denom)

      state(s"s${i}") = _s // 1st moment variables
      state(s"r${i}") = _r // 2nd moment variables
      state(s"denom${i}") = _denom // 3nd moment variables

      i += 1
    }

    (parameter, Array(fx))
  }

  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
                        parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    throw new UnsupportedOperationException("Please use" +
      "optimize(feval: (Array[Tensor[T]]) => (Array[T], Array[Tensor[T]]), parameter: Array[Tensor[T]])" +
      "instead")
  }
}

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

import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.SGDRef

import scala.reflect.ClassTag

/**
 * Implements BERT version of Adam algorithm
 * @param learningRate learning rate
 * @param learningRateDecay learning rate decay
 * @param beta1 first moment coefficient
 * @param beta2 second moment coefficient
 * @param epsilon for numerical stability
 * @tparam T
 */
class BERTAdam[@specialized(Float, Double) T: ClassTag](
   var learningRate: Double = 1e-3,
   var warmupPortion: Double = -1,
   var total: Int = -1,
   var schedule: String = "linear",
   var beta1: Double = 0.9,
   var beta2: Double = 0.999,
   var epsilon: Double = 1e-6,
   var weightDecay: Double = 0.01)(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  @transient
  private var buffer: Tensor[T] = null

  def warmupMethod(x: Double, warmup: Double =0.002): Double = {
    if (x < warmup) {
      x / warmup
    } else if (schedule.equalsIgnoreCase("cosine")) {
      0.5 * (1.0 + math.cos(math.Pi * x))
    } else if (schedule.equalsIgnoreCase("constant")) {
      1.0
    } else if (schedule.equalsIgnoreCase("linear")) {
      1.0 - x
    } else {
      throw new UnsupportedOperationException("Only support cosine|constant|linear schedules")
    }
  }

  /**
   *
   * @param feval     a function that takes a single input (X), the point of a evaluation, and
   *                  returns f(X) and df/dX
   * @param parameter the initial point
   * @return the new x vector and the function list {fx}, evaluated before the update
   */
  override def optimize(feval: (Tensor[T]) => (T, Tensor[T]),
                        parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    if (buffer == null) buffer = Tensor[T]()
    val lr = this.learningRate
    val beta1 = this.beta1
    val beta2 = this.beta2
    val eps = this.epsilon

    val (fx, dfdx) = feval(parameter)

    val state = SGDRef.getstate(this)
    var timestep = state.getOrElse[Double]("evalCounter", 0.0)

    if (!state.get[Tensor[T]]("s").isDefined) {
      state("s") = Tensor[T]().resizeAs(dfdx).zero()
      state("r") = Tensor[T]().resizeAs(dfdx).zero()
      state("denom") = Tensor[T]().resizeAs(dfdx).zero()
    }

    val (_s, _r, _denom) = (state.get[Tensor[T]]("s").get, state.get[Tensor[T]]("r").get,
      state.get[Tensor[T]]("denom").get.resizeAs(dfdx))

    // Optimizer has to set L2 norm clipping


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

    val update = _s.clone().div(_denom)

    // TODO: DON'T DECAY FOR BIAS AND LAYNORM
    if(weightDecay > 0)
    // update += group['weight_decay'] * p.data
      update.add(parameter.clone().mul(ev.fromType(weightDecay)))

    // TODO: make update currentLR configurable
    //lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
//    val currentLR = learningRate * warmupMethod(timestep / total, warmupPortion, "linear")
    val currentLR = updateHyperParameter(timestep)
    val lrScheduled = if(total != -1) {
      // schedule_fct = SCHEDULES[group['schedule']]
      // lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
      currentLR.toDouble * warmupMethod(timestep / total, warmupPortion)
    } else currentLR

    // update_with_lr = lr_scheduled * update
    // p.data.add_(-update_with_lr)
    val updateLR = update.mul(ev.fromType(lrScheduled))
    parameter.add(-updateLR)

    timestep = timestep + 1
    state("evalCounter") = timestep // A tmp tensor to hold the sqrt(v) + epsilon
    state("s") = _s // 1st moment variables
    state("r") = _r // 2nd moment variables
    state("denom") = _denom // 3nd moment variables

    (parameter, Array(fx))
  }

  def updateHyperParameter(timeStep: Double): Double = {
    learningRate * warmupMethod(timeStep / total, warmupPortion)
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.warmupPortion = config.get[Double]("warmupPortion").getOrElse(this.warmupPortion)
    this.total = config.get[Int]("total").getOrElse(this.total)
    this.schedule = config.get[String]("schedule").getOrElse(this.schedule)
    this.beta1 = config.get[Double]("beta1").getOrElse(this.beta1)
    this.beta2 = config.get[Double]("beta2").getOrElse(this.beta2)
    this.epsilon = config.get[Double]("Epsilon").getOrElse(this.epsilon)
    this.weightDecay = config.get[Double]("weightDecay")
      .getOrElse(this.weightDecay)
    this
  }

  override def clearHistory(): Unit = {
    val state = SGDRef.getstate(this)
    state.delete("s")
    state.delete("r")
  }

  override def getLearningRate(): Double = this.learningRate
}

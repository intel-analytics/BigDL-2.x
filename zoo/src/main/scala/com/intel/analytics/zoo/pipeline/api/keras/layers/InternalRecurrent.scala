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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[zoo]class InternalRecurrent[T: ClassTag](
    batchNormParams: BatchNormParams[T] = null,
    maskZero: Boolean = false
)(implicit ev: TensorNumeric[T]) extends Recurrent[T](batchNormParams, maskZero) {

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    require(module.isInstanceOf[Cell[T]],
      "Recurrent: added module should be Cell type!")
    require(!module.isInstanceOf[MultiRNNCell[T]],
      "Recurrent: added module cannot be MultiRNNCell," +
        "use Sequential().add(Recurrent(cell)).add(Recurrent(cell))... instead!")

    topology = module.asInstanceOf[Cell[T]]
    if (topology.preTopology != null) {
      val tmp = topology.preTopology.cloneModule()
      topology.preTopology = null
      preTopology = TimeDistributed(tmp,
        maskZero = maskZero).asInstanceOf[AbstractModule[Activity, Activity, T]]
    }

    if (batchNormParams != null && preTopology == null) {
      throw new IllegalArgumentException(
        s"${topology.getName} does not support BatchNormalization." +
          s" Please add preTopology for it. You can simply using: " +
          s"override def preTopology: AbstractModule[Activity, Activity, T] = Identity()")
    }


    if (batchNormParams != null) {
      val clz =  classOf[Recurrent[T]]
      val methods = clz.getDeclaredMethods.filter(_.getName() == "batchNormalization")
      require(methods.length == 1)
      val method = methods(0)
      method.setAccessible(true)
      val batchNorm= method.invoke(this, batchNormParams).asInstanceOf[TimeDistributed[T]]

      val field = clz.getDeclaredField("layer")
      field.setAccessible(true)
      field.set(this, batchNorm)
      val layer = field.get(this.asInstanceOf[Recurrent[T]]).asInstanceOf[TensorModule[T]]

      preTopology = Sequential[T]().add(preTopology).add(layer)
    }

    if (preTopology != null) {
      modules += preTopology
    }

    modules += topology

    require((preTopology == null && modules.length == 1) ||
      (topology != null && preTopology != null && modules.length == 2),
      "Recurrent extend: should contain only one cell or plus a pre-topology" +
        " to process input")
    this
  }
}

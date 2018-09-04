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


package com.intel.analytics.zoo.pipeline.api.autograd

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable, TensorModule}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.nn.tf.InternalWithoutInput
import com.intel.analytics.bigdl.nn.{InitializationMethod, RandomUniform, VariableFormat, Input => TInput}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, Shape}
import com.intel.analytics.zoo.pipeline.api.Net

import scala.reflect.ClassTag

private[zoo] class KerasParameter[T: ClassTag] private[zoo](val inputShape: Shape,
    val initMethod: InitializationMethod = RandomUniform(-0.05, 0.05),
    val initWeight: Tensor[T] = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](inputShape)
    with InternalWithoutInput with Net {

  if (this.modules.isEmpty == null) {
    build(inputShape)
  }

  def setWeight(tensor: Tensor[T]): Unit = {
    this.labor.asInstanceOf[InternalParameter[T]].setWeight(tensor)
  }

  def getWeight(): Tensor[T] = {
    this.labor.asInstanceOf[InternalParameter[T]].weight
  }

  override def computeOutputShape(inputShape: Shape): Shape = inputShape

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] =
    new InternalParameter[T](
      shape = inputShape,
      initMethod = this.initMethod,
      initWeight = this.initWeight).asInstanceOf[AbstractModule[Activity, Activity, T]]

  override def skipDuplicateCheck(): Boolean = true
}

/**
 * Parameters is a trainable Variable
 */
class Parameter[T: ClassTag] private[zoo](val inputShape: Shape,
    val initMethod: InitializationMethod = RandomUniform(-0.5, 0.5),
    val initWeight: Tensor[T] = null,
    name: String = null)(
    implicit ev: TensorNumeric[T])
  extends Variable[T](null, name) {

  // build and init the KerasParameter
  val kerasParameter = new KerasParameter[T](inputShape, initMethod, initWeight)
  this.node = new Node(kerasParameter)
  this.node.element.asInstanceOf[KerasParameter[T]].build(inputShape)

  def setWeight(tensor: Tensor[T]): Unit = {
    this.node.element.asInstanceOf[KerasParameter[T]].setWeight(tensor)
  }

  def getWeight(): Tensor[T] = {
    this.node.element.asInstanceOf[KerasParameter[T]].getWeight()
  }
}

object Parameter {
  /**
   * Create a trainable Variable
   */
  def apply[T: ClassTag](
      inputShape: Shape,
      initMethod: InitializationMethod = RandomUniform(-0.05, 0.05),
      initWeight: Tensor[T] = null,
      name: String = null)(implicit ev: TensorNumeric[T]): Parameter[T] = {
    new Parameter[T](inputShape, initMethod = initMethod, initWeight = initWeight, name = name)
  }
}

private[zoo] class InternalParameter[T: ClassTag](
    val shape: Shape,
    val initMethod: InitializationMethod = RandomUniform(-0.05, 0.05),
    val initWeight: Tensor[T] = null)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] with Initializable {

  var weight: Tensor[T] =
    if (initWeight != null) initWeight else Tensor[T](shape.toSingle().toArray)

  val gradWeight: Tensor[T] = Tensor[T]()

  setInitMethod(weightInitMethod = initMethod)

  def setWeight(tensor: Tensor[T]): Unit = {
    this.weight = tensor
  }

  override def reset(): Unit = {
    if (initWeight == null) {
      weightInitMethod.init(weight, VariableFormat.OUT_IN)
    }
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(weight)
    output.copy(weight)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput)
    gradInput.copy(gradOutput)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    gradWeight.resizeAs(gradOutput)
    gradWeight.copy(gradOutput)
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def equals(other: Any): Boolean = {
    if (!other.isInstanceOf[InternalParameter[_]]) return false
    this.eq(other.asInstanceOf[InternalParameter[_]])
  }

  override def hashCode(): Int = System.identityHashCode(this)
}

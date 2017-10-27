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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

@SerialVersionUID( 5237686508074490666L)
class RnnCellDS[T : ClassTag](
  inputSize: Int = 4,
  hiddenSize: Int = 3,
  activation: TensorModule[T],
  private var initMethod: InitializationMethod = Default)
                             (implicit ev: TensorNumeric[T])
  extends Cell[T](Array(hiddenSize)) {

  val parallelTable = ParallelTable[T]()
  val i2h = Identity[T]()
  val h2h = Linear[T](hiddenSize, hiddenSize)
  parallelTable.add(i2h)
  parallelTable.add(h2h)
  val cAddTable = CAddTable[T]()

  var cell: AbstractModule[Activity, Activity, T] =
    Sequential[T]()
      .add(parallelTable)
      .add(cAddTable)
      .add(activation)
      .add(ConcatTable()
        .add(Identity[T]())
        .add(Identity[T]()))

  def setInitMethod(initMethod: InitializationMethod): this.type = {
    this.initMethod = initMethod
    this
  }

  override def reset(): Unit = {
    initMethod match {
      case Default =>
        parallelTable.modules.foreach( m => {
          val inputSize = m.asInstanceOf[Linear[T]].weight.size(1).toFloat
          val outputSize = m.asInstanceOf[Linear[T]].weight.size(2).toFloat
          val stdv = 6.0 / (inputSize + outputSize)
          m.asInstanceOf[Linear[T]].weight.apply1( _ =>
            ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
          m.asInstanceOf[Linear[T]].bias.apply1( _ => ev.fromType[Double](0.0))
        })
      case _ =>
        throw new IllegalArgumentException(s"Unsupported initMethod type ${initMethod}")
    }
    zeroGradParameters()
  }

  override def updateOutput(input: Table): Table = {
    output = cell.forward(input).toTable
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = cell.backward(input, gradOutput).toTable
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table,
                                 scale: Double = 1.0): Unit = {
    cell.accGradParameters(input, gradOutput, scale)
  }

  override def updateParameters(learningRate: T): Unit = {
    cell.updateParameters(learningRate)
  }

  override def zeroGradParameters(): Unit = {
    cell.zeroGradParameters()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    cell.parameters()
  }

  override def getParametersTable(): Table = {
    cell.getParametersTable()
  }

  override def toString(): String = {
    val str = "nn.Rnncell"
    str
  }

  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   *
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   *
   * @return
   */
  override def clearState(): RnnCellDS.this.type = {
    super.clearState()
    cell.clearState()
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[RnnCellDS[T]]

  override def equals(other: Any): Boolean = other match {
    case that: RnnCellDS[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        parallelTable == that.parallelTable &&
        i2h == that.i2h &&
        h2h == that.h2h &&
        cAddTable == that.cAddTable &&
        cell == that.cell
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), parallelTable, i2h, h2h, cAddTable, cell)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }

  override def getTimes():
  Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    val times = this.cell.asInstanceOf[Sequential[T]].modules.flatMap(_.getTimes()).toArray
    times
  }
}

object RnnCellDS {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    hiddenSize: Int = 3,
    activation: TensorModule[T])
   (implicit ev: TensorNumeric[T]) : RnnCellDS[T] = {
    new RnnCellDS[T](inputSize, hiddenSize, activation)
  }
}

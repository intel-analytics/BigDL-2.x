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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DataConverter, ModuleData, ModuleSerializable, ModuleSerializer}
import com.intel.analytics.bigdl.utils.{T, Table}
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * The Cell class is a super class of any recurrent kernels, such as
 * [[RnnCell]], [[LSTM]] and [[GRU]]. All the kernels in a recurrent
 * network should extend the [[Cell]] abstract class.
 *
 * @param hiddensShape represents the shape of hiddens which would be
 *                     transferred to the next recurrent time step.
 *                     E.g. For RnnCell, it should be Array(hiddenSize)
 *                     For LSTM, it should be Array(hiddenSize, hiddenSize)
 *                     (because each time step a LSTM return two hiddens `h` and `c` in order,
 *                     which have the same size.)
 *
 *@param regularizers If the subclass has regularizers, it need to put the regularizers into
 *                     an array and pass the array into the [[Cell]] constructor as an argument. See
 *                     [[LSTM]] as a concrete example.
 */
abstract class Cell[T : ClassTag](
  val hiddensShape: Array[Int],
  var regularizers: Array[Regularizer[T]] = null
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Table, T] {

  var subModules: Array[AbstractModule[_ <: Activity, _ <: Activity, T]] = null
  var forwardTimes: Array[Long] = null
  var backwardTimes: Array[Long] = null
  var times: Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = null


  /**
   * Any recurrent kernels should have a cell member variable which
   * represents the module in the kernel.
   *
   * The `cell` receive an input with a format of T(`input`, `preHiddens`), and
   * the output should be a format of T(`output`, `hiddens`).
   * The `hiddens` represents the kernel's output hiddens at the current time step, which will
   * be transferred to next time step. For instance, a simple [[RnnCell]], `hiddens` is h,
   * for LSTM, `hiddens` is T(h, c), and for both of them, the `output` variable represents h.
   * Similarly the `preHiddens` is the kernel's output hiddens at the previous time step.
   *
   */
  var cell: AbstractModule[Activity, Activity, T]

  /**
   * The preTopology defines operations to pre-process the input when it is not dependent
   * on the time dimension. For example, the i2h in SimpleRNN Cell can be calculated before
   * the recurrence since all the input slices are independent.
   *
   * This is particular useful to boost the performance of the recurrent layer.
   *
   * Please define your own preTopology according to your Cell structure.
   * Please refer to SimpleRNN or LSTM for reference.
   * @return
   */
  def preTopology: AbstractModule[Activity, Activity, T] = null

  def hiddenSizeOfPreTopo: Int = hiddensShape(0)

  /**
   * resize the hidden parameters wrt the batch size, hiddens shapes.
   *
   * e.g. RnnCell contains 1 hidden parameter (H), thus it will return Tensor(size)
   *      LSTM contains 2 hidden parameters (C and H) and will return T(Tensor(), Tensor())\
   *      and recursively intialize all the tensors in the Table.
   *
   * @param hidden
   * @param batchSize batchSize
   * @return
   */
  def hidResize(hidden: Activity, batchSize: Int, imageSize: Array[Int] = null): Activity = {
    if (hidden == null) {
      if (hiddensShape.length == 1) {
        hidResize(Tensor[T](), batchSize)
      } else {
        val _hidden = T()
        var i = 1
        while (i <= hiddensShape.length) {
          _hidden(i) = Tensor[T]()
          i += 1
        }
        hidResize(_hidden, batchSize, imageSize)
      }
    } else {
      if (hidden.isInstanceOf[Tensor[T]]) {
        require(hidden.isInstanceOf[Tensor[T]],
          "Cell: hidden should be a Tensor")
        hidden.toTensor.resize(batchSize, hiddensShape(0))
      } else {
        require(hidden.isInstanceOf[Table],
          "Cell: hidden should be a Table")
        var i = 1
        if (null == imageSize) {
          while (i <= hidden.toTable.length()) {
            hidden.toTable[Tensor[T]](i).resize(batchSize, hiddensShape(i - 1))
            i += 1
          }
        } else {
          val sizes = new Array[Int](imageSize.length + 2)
          sizes(0) = batchSize
          Array.copy(imageSize, 0, sizes, 2, imageSize.size)
          while (i <= hidden.toTable.length()) {
            sizes(1) = hiddensShape(i - 1)
            hidden.toTable[Tensor[T]](i).resize(sizes)
            i += 1
          }
        }
        hidden
      }
    }
  }

  override def updateOutput(input: Table): Table = {
    output = cell.forward(input).toTable
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    gradInput = cell.updateGradInput(input, gradOutput).toTable
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    cell.accGradParameters(input, gradOutput)
  }

  override def backward(input: Table, gradOutput: Table): Table = {
    gradInput = cell.backward(input, gradOutput).toTable
    gradInput
  }

  override def updateParameters(learningRate: T): Unit = {
    cell.updateParameters(learningRate)
  }

  private def initAddTimes(): Unit = {
    val cellTimes = cell.getTimes
    if (subModules == null || subModules.length < cellTimes.length) {
      subModules = new Array[AbstractModule[_ <: Activity, _ <: Activity, T]](cellTimes.length)
      var i = 0
      while (i < cellTimes.length) {
        subModules(i) = cellTimes(i)._1
        i += 1
      }
      forwardTimes = new Array[Long](cellTimes.length)
      backwardTimes = new Array[Long](cellTimes.length)
      times =
        new Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)](cellTimes.length)
    }
  }

  private def resetAddTimes(): Unit = {
    if (subModules != null) {
      var i = 0
      while (i < subModules.length) {
        forwardTimes(i) = 0L
        backwardTimes(i) = 0L
        i += 1
      }
    }
  }

  def addTimes(other: Cell[T]): Unit = {
    val cellTimes = cell.getTimes
    val otherTimes = other.getTimes
    require(cellTimes.length == otherTimes.length,
      " Cell -> CellTimes: cell.getTimes.length does not comform to other.getTimes.length." +
        s" cell.getTimes.length = ${cellTimes.length}, " +
        s"other.getTimes.length = ${otherTimes.length}")

    val length = cellTimes.length
    initAddTimes()
    var i = 0
    while (i < length) {
      val subModule = otherTimes(i)._1.getClass.getName
      require(subModules(i).getClass.getName == subModule,
        s"Cell -> CellTimes: ${i}-th submodule in cell" +
          s" does not comform to ${i}-th submodule in other." +
          s" ${i}-th cell module is ${subModules(i)}," +
          s" ${i}-th other module is ${otherTimes(i)._1}")
      forwardTimes(i) += otherTimes(i)._2
      backwardTimes(i) += otherTimes(i)._3
      i += 1
    }
  }

  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    initAddTimes()
    val cellTimes = cell.getTimes
    var i = 0
    while (i < cellTimes.length) {
      times(i) = (subModules(i),
        forwardTimes(i) + cellTimes(i)._2,
        backwardTimes(i) + cellTimes(i)._3)
      i += 1
    }
    times
  }

  override def resetTimes(): Unit = {
    resetAddTimes()
    cell.resetTimes
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

  override def reset(): Unit = {
    cell.reset()
  }

  /**
   * Use this method to set the whether the recurrent cell
   * is regularized
   *
   * @param isRegularized whether to be regularized or not
   */
  def regluarized(
    isRegularized: Boolean
  ): Unit = {
    if (null != regularizers) {
      regularizers.foreach(x =>
        if (null != x) {
          if (isRegularized) x.enable()
          else x.disable()
        }
      )
    }
  }
}

object CellSerializer extends ModuleSerializable {

  override def doLoadModule[T: ClassTag](model : BigDLModule)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val module = super.doLoadModule(model)
    val cellModule = module.asInstanceOf[Cell[T]]

    val attrMap = model.getAttrMap
    cellModule.cell = DataConverter.getAttributeValue(attrMap.get("cell")).
      asInstanceOf[AbstractModule[Activity, Activity, T]]

    cellModule
  }

  override def doSerializeModule[T: ClassTag](module : ModuleData[T],
                                              cellModuleBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(module, cellModuleBuilder)
    val cellModule = module.module.asInstanceOf[Cell[T]]

    val cellSerializerFlagBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(cellSerializerFlagBuilder, true,
      scala.reflect.runtime.universe.typeOf[Boolean])
    cellModuleBuilder.putAttr("is_cell_module", cellSerializerFlagBuilder.build)

    val cellBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(cellBuilder, cellModule.cell,
      ModuleSerializer.abstractModuleType)
    cellModuleBuilder.putAttr("cell", cellBuilder.build)

  }
}

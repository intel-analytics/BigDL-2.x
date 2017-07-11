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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Gated Recurrent Units architecture.
 * The first input in sequence uses zero value for cell and hidden state
 *
 * Ref.
 * 1. http://www.wildml.com/2015/10/
 * recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
 *
 * 2. https://github.com/Element-Research/rnn/blob/master/GRU.lua
 *
 * @param inputSize the size of each input vector
 * @param outputSize Hidden unit size in GRU
 * @param  p is used for [[Dropout]] probability. For more details about
 *           RNN dropouts, please refer to
 *           [RnnDrop: A Novel Dropout for RNNs in ASR]
 *           (http://www.stat.berkeley.edu/~tsmoon/files/Conference/asru2015.pdf)
 *           [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks]
 *           (https://arxiv.org/pdf/1512.05287.pdf)
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param uRegularizer: instance [[Regularizer]]
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
            applied to the bias.
 */
@SerialVersionUID(6717988395573528459L)
class GRU[T : ClassTag] (
  val inputSize: Int,
  val outputSize: Int,
  val p: Double = 0,
  var wRegularizer: Regularizer[T] = null,
  var uRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null)(implicit ev: TensorNumeric[T])
  extends Cell[T](
    hiddensShape = Array(outputSize),
    regularizers = Array(wRegularizer, uRegularizer, bRegularizer)
  ) {
  var i2g: AbstractModule[_, _, T] = _
  var h2g: AbstractModule[_, _, T] = _
  var gates: AbstractModule[_, _, T] = _
  val featDim = 2
  override var cell: AbstractModule[Activity, Activity, T] = buildGRU()

  override def preTopology: AbstractModule[Activity, Activity, T] =
    if (p != 0) {
      null
    } else {
      TimeDistributed[T](Linear(inputSize, 3 * outputSize,
        wRegularizer = wRegularizer, bRegularizer = bRegularizer))
    }

  def buildGates(): AbstractModule[Activity, Activity, T] = {
    if (p != 0) {
      i2g = Sequential()
        .add(ConcatTable()
          .add(Dropout(p))
          .add(Dropout(p)))
        .add(ParallelTable()
          .add(Linear(inputSize, outputSize,
            wRegularizer = wRegularizer, bRegularizer = bRegularizer))
          .add(Linear(inputSize, outputSize,
            wRegularizer = wRegularizer, bRegularizer = bRegularizer)))
        .add(JoinTable(2, 0))

      h2g = Sequential()
        .add(ConcatTable()
          .add(Dropout(p))
          .add(Dropout(p)))
        .add(ParallelTable()
          .add(Linear(outputSize, outputSize, withBias = false,
            wRegularizer = uRegularizer))
          .add(Linear(outputSize, outputSize, withBias = false,
            wRegularizer = uRegularizer)))
        .add(JoinTable(2, 0))
    } else {
      i2g = Narrow[T](featDim, 1, 2 * outputSize)
      h2g = Linear(outputSize, 2 * outputSize, withBias = false,
        wRegularizer = uRegularizer)
    }

    gates = Sequential()
      .add(ParallelTable()
        .add(i2g)
        .add(h2g))
      .add(CAddTable(true))
      .add(Reshape(Array(2, outputSize)))
      .add(SplitTable(1, 2))
      .add(ParallelTable()
        .add(Sigmoid())
        .add(Sigmoid()))

    gates
  }

  def buildGRU(): AbstractModule[Activity, Activity, T] = {
    buildGates()

    val gru = Sequential()
      .add(ConcatTable()
        .add(Identity())
        .add(gates))
      .add(FlattenTable()) // x(t), h(t - 1), r(t), z(t)

    val f2g = if (p != 0) {
      Sequential()
        .add(Dropout(p))
        .add(Linear(inputSize, outputSize,
            wRegularizer = wRegularizer, bRegularizer = bRegularizer))
    } else {
      Narrow(featDim, 1 + 2 * outputSize, outputSize)
    }

    val h_hat = Sequential()
      .add(ConcatTable()
        .add(Sequential()
          .add(SelectTable(1))
          .add(f2g))
        .add(Sequential()
        .add(NarrowTable(2, 2))
        .add(CMulTable())))
      .add(ParallelTable()
        .add(Identity())
        .add(Sequential()
         .add(Dropout(p))
         .add(Linear(outputSize, outputSize, withBias = false,
           wRegularizer = uRegularizer))))
      .add(CAddTable(true))
      .add(Tanh())

    gru
      .add(ConcatTable()
        .add(Sequential()
          .add(ConcatTable()
            .add(h_hat)
            .add(Sequential()
              .add(SelectTable(4))
              .add(MulConstant(-1))
              .add(AddConstant(1))))
          .add(CMulTable()))
        .add(Sequential()
          .add(ConcatTable()
            .add(SelectTable(2))
            .add(SelectTable(4)))
          .add(CMulTable())))
      .add(CAddTable(false))
      .add(ConcatTable()
        .add(Identity[T]())
        .add(Identity[T]()))

    cell = gru
    cell
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[GRU[T]]

  override def equals(other: Any): Boolean = other match {
    case that: GRU[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        inputSize == that.inputSize &&
        outputSize == that.outputSize &&
        p == that.p
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), inputSize, outputSize, p)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object GRU {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int = 4,
    outputSize: Int = 3,
    p: Double = 0,
    wRegularizer: Regularizer[T] = null,
    uRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null)(implicit ev: TensorNumeric[T]): GRU[T] = {
    new GRU[T](inputSize, outputSize, p, wRegularizer, uRegularizer, bRegularizer)
  }
}

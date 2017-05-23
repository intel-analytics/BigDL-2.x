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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

@SerialVersionUID( - 643193217505024792L)
class BiRecurrentDS[T : ClassTag](
    merge: AbstractModule[Table, Tensor[T], T] = null,
    isCloneInput: Boolean = true) (implicit ev: TensorNumeric[T])
  extends Container[Tensor[T], Tensor[T], T] {

  val timeDim = 2
  val featDim = 3
  val layer: Recurrent[T] = Recurrent[T]()
  val revLayer: Recurrent[T] = Recurrent[T]()
  val birnn = Sequential[T]()

  if (isCloneInput) {
    birnn.add(ConcatTable()
      .add(Identity[T]())
      .add(Identity[T]()))
  } else {
    birnn.add(BifurcateSplitTable[T](featDim))
  }

  birnn
    .add(ParallelTable[T]()
      .add(layer)
      .add(Sequential[T]()
        .add(Reverse[T](timeDim))
        .add(revLayer)
        .add(ReverseDS[T](timeDim))))
  if (merge == null) birnn.add(CAddTable[T](true))
  else birnn.add(merge)

  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]):
    BiRecurrentDS.this.type = {
    layer.add(module)
    revLayer.add(module.cloneModule())
    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = birnn.updateOutput(input).toTensor[T]
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    birnn.accGradParameters(input, gradOutput, scale)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = birnn.updateGradInput(input, gradOutput).toTensor[T]
    gradInput
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = birnn.parameters()

  override def updateParameters(learningRate: T): Unit = birnn.updateParameters(learningRate)

  /**
   * If the module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  override def zeroGradParameters(): Unit = birnn.zeroGradParameters()

  override def training(): BiRecurrentDS.this.type = {
    super.training()
    birnn.training()
    this
  }

  override def evaluate(): BiRecurrentDS.this.type = {
    super.evaluate()
    birnn.evaluate()
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[BiRecurrentDS[T]]

  /**
   * Clear cached activities to save storage space or network bandwidth. Note that we use
   * Tensor.set to keep some information like tensor share
   *
   * The subclass should override this method if it allocate some extra resource, and call the
   * super.clearState in the override method
   *
   * @return
   */
  override def clearState(): BiRecurrentDS.this.type = {
    birnn.clearState()
    this
  }

  override def toString(): String = s"BiRecurrentDS($timeDim, $birnn)"

  override def equals(other: Any): Boolean = other match {
    case that: BiRecurrentDS[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        timeDim == that.timeDim &&
        layer == that.layer &&
        revLayer == that.revLayer &&
        birnn == that.birnn
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(), timeDim, layer, revLayer, birnn)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object BiRecurrentDS {
  def apply[@specialized(Float, Double) T: ClassTag](
    merge: AbstractModule[Table, Tensor[T], T] = null, isCloneInput: Boolean = true)
    (implicit ev: TensorNumeric[T]) : BiRecurrentDS[T] = {
    new BiRecurrentDS[T](merge, isCloneInput)
  }
}

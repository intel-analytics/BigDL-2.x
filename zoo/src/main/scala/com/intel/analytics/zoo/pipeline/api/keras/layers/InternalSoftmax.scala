package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, Shape}

import scala.concurrent.Future
import scala.reflect.ClassTag

class InternalSoftmax[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    val sizes = input.size()
    val shift = input.max(dim)._1

    val shiftedInput = input.sub(shift.expand(sizes))
    val exp = shiftedInput.exp()

    val sum = exp.sum(dim)
    output = exp.div(sum.expand(sizes))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dim = input.dim()
    val sum = (output.clone().cmul(gradOutput)).sum(dim)
    gradInput = output.clone().cmul(gradOutput - sum.expand(input.size()))
    gradInput
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
}

object InternalSoftmax{

  def apply[@specialized(Float, Double) T: ClassTag]()
                                                    (implicit ev: TensorNumeric[T]) : InternalSoftmax[T] = {
    new InternalSoftmax[T]()
  }
}

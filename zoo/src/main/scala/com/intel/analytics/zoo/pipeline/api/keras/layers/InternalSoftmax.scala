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

    val input2 = input.clone()
    val shiftedInput = input2.sub(shift.expand(sizes))
    val exp = shiftedInput.exp()

    val sum = exp.sum(dim)
    output = exp.div(sum.expand(sizes))
    output
  }

  @transient
  private var results: Array[Future[Unit]] = null

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(output)
    val (nFrame, stride) = if (input.nDimension() == 1) {
      (1, 1)
    } else if (input.nDimension() == 2) {
      (input.size(1), 1)
    } else if (input.nDimension() == 3) {
      (1, input.size(2) * input.size(3))
    } else {
      (input.size(1), input.size(3) * input.size(4))
    }
    if (results == null || results.length != nFrame * stride) {
      results = new Array[Future[Unit]](nFrame * stride)
    }
    SoftMax.updateGradInput[T](input, gradOutput, gradInput, output, results)
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.Mean
import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.autograd.AutoGrad

import scala.reflect.ClassTag

class InternalLayerNorm[T: ClassTag]()
  (implicit ev: TensorNumeric[T]) extends TensorModule[T]{
  val hiddenSize = 768
  val dim = 3
  val weight: Tensor[T] = Tensor.ones[T](hiddenSize).view(1, hiddenSize)
  val bias: Tensor[T] = Tensor[T](hiddenSize).view(1, hiddenSize)

  val gradWeight: Tensor[T] = Tensor[T]()
  val gradBias: Tensor[T] = Tensor[T]()

  val meanLayer = Mean[T](dimension = dim,
    squeeze = false)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
//    val sizes = x.getOutputShape().toSingle().toArray
//    val u = AutoGrad.mean(x, sizes.size - 1, true)
//    val t = x - u
//    val s = AutoGrad.mean(AutoGrad.square(t), sizes.size -1, true)
//    val y = (t) / AutoGrad.sqrt(s + e)
//    y * weight + bias

    val u = meanLayer.forward(input).toTensor[T]
    val t = input.clone().sub(u)
    val s = meanLayer.forward(t.clone().square()).toTensor[T]
    val y = t.div(s.add(ev.fromType(1e-5)).sqrt())
    output = y.cmul(weight).add(bias)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }
}

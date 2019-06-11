package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.{Mean, Sum}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class InternalLayerNorm[T: ClassTag](val nOutput: Int = 768, val eps: Double = 1e-5)
  (implicit ev: TensorNumeric[T]) extends TensorModule[T]{
  val weight: Tensor[T] = Tensor.ones[T](nOutput).view(1, nOutput)
  val bias: Tensor[T] = Tensor[T](nOutput).view(1, nOutput)

  var gradWeight: Tensor[T] = Tensor[T]()
  var gradBias: Tensor[T] = Tensor[T]()

  var inputDim: Int = 0
  var meanLayer: TensorModule[T] = null
  var y: Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (inputDim != input.dim()) {
      inputDim = input.dim()
      meanLayer = Mean[T](dimension = inputDim, squeeze = false)
    }
    val dim = input.dim()
//    val u = meanLayer.forward(input).toTensor[T]
    val u = input.sum(dim).div(ev.fromType(input.size(dim)))
    val t = input.clone().sub(u)
    val s = meanLayer.forward(t.clone().square()).toTensor[T]
    y = t.div(s.add(ev.fromType(eps)).sqrt())
    output = y.clone().cmul(weight).add(bias)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    var i = 1
    gradWeight = y.clone().cmul(gradOutput)
    gradBias = gradOutput
    while (i < gradOutput.dim()) {
      gradBias = gradBias.sum(i)
      gradWeight = gradWeight.sum(i)
      i += 1
    }
    gradBias.resize(bias.size())
    gradWeight.resize(weight.size())
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }
}

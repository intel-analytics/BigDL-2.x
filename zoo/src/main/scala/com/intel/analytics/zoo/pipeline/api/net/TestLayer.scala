package com.intel.analytics.zoo.pipeline.api.net

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.pipeline.api.Predictable

import scala.reflect.ClassTag

/**
  * Created by yuhao on 7/1/19.
  */
class TestLayer extends AbstractModule[Tensor[Float], Tensor[Float], Float] with Predictable[Float]{

  protected val module: Module[Float] = this
  implicit val ev = TensorNumeric.NumericFloat
  implicit val tag: ClassTag[Float] = ClassTag.Float

  var weights = Tensor.ones(2)

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    val gradients = Tensor.ones[Float](2) / 10
    println(this + "-------" + "weights: " + weights.storage().array().mkString(", ") +
      "-----" + "gradients: " + gradients.storage().array().mkString(", "))
    (Array(weights), Array(gradients))
  }

  override def updateOutput(input: Tensor[Float]): Tensor[Float] = {
    output = Tensor.ones(1)
    output
  }

  override def updateGradInput(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    gradInput = gradOutput
    gradInput
  }

}

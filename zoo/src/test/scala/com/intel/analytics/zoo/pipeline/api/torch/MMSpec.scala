package com.intel.analytics.zoo.pipeline.api.torch

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.zoo.pipeline.api.autograd.AutoGrad



class MMSpec extends FlatSpec with Matchers {

  "mm" should "be ok" in {
    val input1 = Variable[Float](inputShape = Shape(4, 2))
    val input2 = Variable[Float](inputShape = Shape(4, 2))
    val result = AutoGrad.mm(input1, input2, axes = List(2, 2))
    val model = Model[Float](input = Array(input1, input2), output = result)
    val recordNum = 2
    val i1 = Tensor[Float](recordNum, 3, 4).rand()
    val i2 = Tensor[Float](recordNum, 3, 4).rand()
    val o1 = model.forward(T(i1, i2)).toTensor[Float].clone()
    val o2 = model.forward(T(i1, i2)).toTensor[Float].clone()
    assert(o1.almostEqual(o2, 1e-5))
  }
}

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential


class GetShapeSpec extends KerasBaseSpec  {

  "Dense" should "be the same as Keras" in {
    val ss = new GetShape[Float](inputShape = Shape(3, 2))
    val seq = Sequential[Float]()
    val input = InputLayer[Float](inputShape = Shape(3, 4), name = "input1")
    seq.add(input)
    val dense = Dense[Float](2, activation = "relu")
    seq.add(dense)
    seq.add(ss)
    seq.getOutputShape().toSingle().toArray should be (Array(2))
    val outShape = seq.forward(Tensor[Float](Array(2, 3, 4)).randn())
    outShape
  }

}

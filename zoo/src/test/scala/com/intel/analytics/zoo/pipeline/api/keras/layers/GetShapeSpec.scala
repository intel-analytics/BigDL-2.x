package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random


class GetShapeSpec extends KerasBaseSpec {

  "GetShape" should "be test" in {
    val seq = Sequential[Float]()
    val input = InputLayer[Float](inputShape = Shape(3, 4), name = "input1")
    seq.add(input)
    val dense = Dense[Float](2, activation = "relu")
    seq.add(dense)
    val ss = new GetShape[Float](inputShape = Shape(3, 2))
    seq.add(ss)
    seq.getOutputShape().toSingle().toArray should be(Array(3))
    val outShape = seq.forward(Tensor[Float](Array(2, 3, 4)).randn())
    outShape.toTensor[Float].storage().toArray should be(Array(2.0, 3.0, 2.0))
  }

  class GetShapeSerialTest extends ModuleSerializationTest {
    override def test(): Unit = {
      val ss = new GetShape[Float](inputShape = Shape(3, 2))
      ss.build(Shape(3, 2))
      val input = Tensor[Float](2, 3, 2).apply1(_ => Random.nextFloat())
      runSerializationTest(ss, input)
    }
  }

}

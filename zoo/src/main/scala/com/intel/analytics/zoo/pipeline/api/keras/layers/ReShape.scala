package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag


class GetShape[T: ClassTag](
    val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape)) {

  override def computeOutputShape(inputShape: Shape): Shape = {
    Shape.apply(Array(inputShape.toSingle().toArray.length))
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    new InternalGetShape[T]()
  }
}

private class InternalGetShape[T: ClassTag](implicit ev: TensorNumeric[T])
  extends AbstractModule[Tensor[T], Tensor[T], T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val shape = input.toTensor[T].size().map(i => ev.fromType[Int](i))
    Tensor(data = shape, shape = Array(shape.length))
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    input.toTensor.clone().fill(ev.zero)
  }
}

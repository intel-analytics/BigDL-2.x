package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.nn.{ErrorInfo, InitializationMethod, Xavier}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape => InternalShape}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag


class GetShape[T: ClassTag](
    val inputShape: InternalShape = null)(implicit ev: TensorNumeric[T])extends KerasLayer[Tensor[T], Tensor[T], T](KerasUtils.addBatch(inputShape))  {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    input
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradOutput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {

  }

  override def computeOutputShape(inputShape: InternalShape): InternalShape = {
    InternalShape.apply(Array(inputShape.toSingle().toArray.length))
  }

  override def doBuild(inputShape: InternalShape): AbstractModule[Tensor[T], Tensor[T], T] = this
}


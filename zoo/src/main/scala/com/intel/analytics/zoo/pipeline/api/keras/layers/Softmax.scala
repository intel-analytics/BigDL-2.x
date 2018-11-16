package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, SoftMax => KSoftMax}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, SingleShape}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.{Transpose, Sequential => TSequential, TimeDistributed => TTimeDistributed}

import scala.reflect.ClassTag

class SoftMax[T: ClassTag](var inputShape2: Shape = null)(implicit ev: TensorNumeric[T])
  extends KSoftMax[T](inputShape2) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val layer = com.intel.analytics.bigdl.nn.SoftMax()
    if (input.length <= 2) {
      layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
    else {
      val seq = TSequential[T]()
      seq.add(Transpose(Array((1, 3))))
      seq.add(layer)
      seq.add(Transpose(Array((1, 3))))

      val model = if (input.length > 3)
        TTimeDistributed[T](seq.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]])
      else seq

      model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
    }
  }
}

object SoftMax {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): SoftMax[T] = {
    new SoftMax[T](inputShape)
  }
}
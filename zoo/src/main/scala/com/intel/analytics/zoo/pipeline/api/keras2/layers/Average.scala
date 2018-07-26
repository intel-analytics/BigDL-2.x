package com.intel.analytics.zoo.pipeline.api.keras2.layers

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.Merge

import scala.reflect.ClassTag

/**
  * Created by xiaxin on 7/24/18.
  */
class Average[T: ClassTag](override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Merge[T](layers = null, mode = "ave", inputShape = inputShape)
    with Net {
}
object Average{
  def apply[@specialized(Float, Double) T: ClassTag]
  (inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Average[T] = {
    new Average[T](inputShape)
  }
  def average[@specialized(Float, Double) T: ClassTag](inputs: List[ModuleNode[T]])
  (implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    val layer = new Average[T]()
    layer.inputs(inputs.toArray)
  }
}
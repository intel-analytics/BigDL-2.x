package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.Reverse
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.{Recurrent => BKerasRecurrent, Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

abstract class Recurrent[T: ClassTag](
    override val outputDim: Int,
    override val returnSequences: Boolean = false,
    override val goBackwards: Boolean = false,
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends BKerasRecurrent[T](outputDim, returnSequences, goBackwards, inputShape) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val model = TSequential[T]()
    if (goBackwards) model.add(Reverse(2))
    val rec = new com.intel.analytics.zoo.pipeline.api.keras.layers.b.TorchRecurrent[T]()
    rec.add(buildCell(input))
    model.add(rec)
    if (!returnSequences) model.add(Select(2, -1))
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}
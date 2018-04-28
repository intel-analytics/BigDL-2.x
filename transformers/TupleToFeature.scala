package com.intel.analytics.zoo.pipeline.nnframes.transformers

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class TupleToFeature[F, T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Transformer[(F, Double), F] {

  override def apply(prev: Iterator[(F, Double)]): Iterator[F] = {
    prev.map(_._1)
  }
}

object TupleToFeature {
  def apply[F, T: ClassTag]()(implicit ev: TensorNumeric[T]) =
    new TupleToFeature()
}


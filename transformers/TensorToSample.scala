package com.intel.analytics.zoo.pipeline.nnframes.transformers

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * a Transformer that converts Tensor to Sample.
 */
class TensorToSample[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Transformer[Tensor[T], Sample[T]] {

  override def apply(prev: Iterator[Tensor[T]]): Iterator[Sample[T]] = {
    prev.map(Sample(_))
  }
}

object TensorToSample {
  def apply[F, T: ClassTag]()(implicit ev: TensorNumeric[T]) =
    new TensorToSample()
}


package com.intel.analytics.zoo.pipeline.nnframes.transformers

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class TensorToSampleAdapter[F, T: ClassTag](
    tensorTransformer: Transformer[F, Tensor[T]])(implicit ev: TensorNumeric[T])
  extends Transformer[F, Sample[T]] {

  override def apply(prev: Iterator[F]): Iterator[Sample[T]] = {
    tensorTransformer.apply(prev).map(Sample(_))
  }
}


object TensorToSampleAdapter {
  def apply[F, T: ClassTag](
      tensorTransformer: Transformer[F, Tensor[T]])(implicit ev: TensorNumeric[T]) =
    new TensorToSampleAdapter(tensorTransformer)
}


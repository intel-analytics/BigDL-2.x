package com.intel.analytics.zoo.pipeline.nnframes.transformers.internal

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class FeatureToTupleAdapter[F, T](
    sampleTransformer: Transformer[(F, Any), Sample[T]]
  )(implicit ev: TensorNumeric[T]) extends Transformer[F, Sample[T]] {

  override def apply(prev: Iterator[F]): Iterator[Sample[T]] = {
    sampleTransformer.apply(prev.map(f => (f, null)))
  }
}

object FeatureToTupleAdapter {
  def apply[F, L, T: ClassTag](
      sampleTransformer: Transformer[(F, Any), Sample[T]])(implicit ev: TensorNumeric[T]) =
    new FeatureToTupleAdapter(sampleTransformer)
}

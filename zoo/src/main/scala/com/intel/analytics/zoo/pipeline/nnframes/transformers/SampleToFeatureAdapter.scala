package com.intel.analytics.zoo.pipeline.nnframes.transformers

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SampleToFeatureAdapter[F, T](
    sampleTransformer: Transformer[(F, Any), Sample[T]]
  )(implicit ev: TensorNumeric[T]) extends Transformer[F, Sample[T]] {

  override def apply(prev: Iterator[F]): Iterator[Sample[T]] = {
    sampleTransformer.apply(prev.map(f => (f, null)))
  }
}

object SampleToFeatureAdapter {
  def apply[F, L, T: ClassTag](
      sampleTransformer: Transformer[(F, Any), Sample[T]])(implicit ev: TensorNumeric[T]) =
    new SampleToFeatureAdapter(sampleTransformer)
}

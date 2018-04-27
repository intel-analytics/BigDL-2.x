package com.intel.analytics.zoo.pipeline.nnframes.transformers

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature

import scala.reflect.ClassTag

class ImageFeatureToTensor [T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Transformer[ImageFeature, Tensor[T]] {

  override def apply(prev: Iterator[ImageFeature]): Iterator[Tensor[T]] = {
    prev.map { imf =>
      imf(ImageFeature.imageTensor).asInstanceOf[Tensor[T]]
    }
  }
}

object ImageFeatureToTensor {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]) = new ImageFeatureToTensor[T]()
}


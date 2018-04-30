package com.intel.analytics.zoo.pipeline.nnframes.transformers

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.pipeline.nnframes.NNImageSchema
import org.apache.spark.sql.Row

import scala.reflect.ClassTag

/**
 * a Transformer that converts a Spark Row to a BigDL ImageFeature.
 */
class RowToImageFeature [T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Transformer[Row, ImageFeature] {

  override def apply(prev: Iterator[Row]): Iterator[ImageFeature] = {
    prev.map { row =>
      NNImageSchema.row2IMF(row)
    }
  }
}

object RowToImageFeature {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]) = new RowToImageFeature[T]()
}
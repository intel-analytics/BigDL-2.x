package com.intel.analytics.zoo.feature.common

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag


/**
 * filter null out of the iterator
 */
class FilterNull[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends Preprocessing[Any, Any] {

  override def apply(prev: Iterator[Any]): Iterator[Any] = {
    prev.filter( _ != null)
  }
}

object FilterNull {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): FilterNull[T] =
    new FilterNull[T]()
}


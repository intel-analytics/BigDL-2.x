package com.intel.analytics.zoo.tfpark

import com.intel.analytics.bigdl.models.utils.{ModelBroadcast, ModelBroadcastFactory, ModelBroadcastImp}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[zoo] class TFModelBroadcastFactory extends ModelBroadcastFactory {
  override def create[T: ClassTag]()(implicit ev: TensorNumeric[T]): ModelBroadcast[T] = {
    new TFModelBroadcastV2[T]()
  }
}


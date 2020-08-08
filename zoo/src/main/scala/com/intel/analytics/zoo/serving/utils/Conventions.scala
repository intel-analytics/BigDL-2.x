package com.intel.analytics.zoo.serving.utils

import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType

object Conventions {
  val SERVING_STREAM_NAME= "serving_stream"
  val ARROW_INT = new ArrowType.Int(32, true)
  val ARROW_FLOAT = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
  val ARROW_BINARY = new ArrowType.Binary()
  val ARROW_UTF8 = new ArrowType.Utf8
}

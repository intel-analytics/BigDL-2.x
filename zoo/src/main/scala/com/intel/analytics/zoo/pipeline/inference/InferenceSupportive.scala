package com.intel.analytics.zoo.pipeline.inference

import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.tensor.Tensor

trait InferenceSupportive {

  val logger = LoggerFactory.getLogger(getClass)

  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    logger.info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }

  def transferInferenceInputToTensor(input: java.util.List[java.util.List[java.lang.Float]]): Tensor[Float] = {
    val arrays = input.asScala.map(_.asScala.toArray.map(_.asInstanceOf[Float]))
    val _buffer = Tensor[Float]()
    arrays.length match {
      case 0 => {}
      case 1 => {
        val size = arrays.head.length
        _buffer.resize(size)
        System.arraycopy(arrays.head, 0, _buffer.storage().array(), 0, size)
      }
      case _ => {
        val size = arrays.head.length
        arrays.map(arr => require(size == arr.length, "input array have different lengths"))
        _buffer.resize(Array(arrays.length, size))
        var d = 0
        while (d < arrays.length) {
          System.arraycopy(arrays(d), 0, _buffer.storage().array(), d * size, size)
          d += 1
        }
      }
    }
    _buffer
  }

}
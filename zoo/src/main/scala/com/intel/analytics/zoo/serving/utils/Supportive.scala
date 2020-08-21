package com.intel.analytics.zoo.serving.utils

import org.apache.log4j.Logger

trait Supportive {
  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    Logger.getLogger(getClass).info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }
}

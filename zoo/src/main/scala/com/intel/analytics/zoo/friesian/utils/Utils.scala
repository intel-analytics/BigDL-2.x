package com.intel.analytics.zoo.friesian.utils

import java.util.{List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.codahale.metrics.Timer
import org.apache.log4j.Logger

import scala.collection.JavaConverters._

object Utils {
  val logger: Logger = Logger.getLogger(getClass)

  def timing[T](name: String)(timers: Timer*)(f: => T): T = {
    val begin = System.nanoTime()
    val contexts = timers.map(_.time())
    val result = f
    contexts.map(_.stop())
    val end = System.nanoTime()
    val cost = (end - begin)
    logger.info(s"$name time elapsed [${cost / 1e6} ms]")
    result
  }

  def silent[T](name: String)(timers: Timer*)(f: => T): T = {
    val contexts = timers.map(_.time())
    val result = f
    contexts.map(_.stop())
    result
  }
}

case class TimerMetrics(name: String,
                        count: Long,
                        meanRate: Double,
                        min: Long,
                        max: Long,
                        mean: Double,
                        median: Double,
                        stdDev: Double,
                        _75thPercentile: Double,
                        _95thPercentile: Double,
                        _98thPercentile: Double,
                        _99thPercentile: Double,
                        _999thPercentile: Double)

object TimerMetrics {
  def apply(name: String, timer: Timer): TimerMetrics =
    TimerMetrics(
      name,
      timer.getCount,
      timer.getMeanRate,
      timer.getSnapshot.getMin / 1000000,
      timer.getSnapshot.getMax / 1000000,
      timer.getSnapshot.getMean / 1000000,
      timer.getSnapshot.getMedian / 1000000,
      timer.getSnapshot.getStdDev / 1000000,
      timer.getSnapshot.get75thPercentile() / 1000000,
      timer.getSnapshot.get95thPercentile() / 1000000,
      timer.getSnapshot.get98thPercentile() / 1000000,
      timer.getSnapshot.get99thPercentile() / 1000000,
      timer.getSnapshot.get999thPercentile() / 1000000
    )
}

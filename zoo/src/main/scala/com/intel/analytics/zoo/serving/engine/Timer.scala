package com.intel.analytics.zoo.serving.engine

import java.util.PriorityQueue

import org.apache.log4j.Logger


class Timer(name: String = "") {
  var total: Float = 0 // total cost up to now
  var record: Int = 0 // total record number up to now
  var batchNum: Int = 0 // total batch number up to now
  var average: Float = 0 // average cost up to now
  var averageBatch: Float = 0 // average cost per batch up to now
  var first: Long = 0 // first record to trigger warm up
  var max: Long = Long.MinValue // max cost up to now
  var min: Long = Long.MaxValue // min cost up to now
  val topQ = new PriorityQueue[Long]()
  val nQ = 10
  topQ.add(Long.MinValue)
  def print(): Unit = {
    println(s"total cost $total, record num $record, average per input $average, " +
      s"average per batch $averageBatch, first $first, max $max, min $min (ms/batch)")
    println(s"Top $nQ of statistic:")
    var tmpArr = Array[Long]()
    (0 until nQ + 1).foreach(i => {
      if (topQ.isEmpty) {
        tmpArr.foreach(ele => topQ.add(ele))
        return
      }
      tmpArr = tmpArr :+ topQ.peek()
      println(s"Top ${nQ - i}: ${topQ.poll()} ms")
    })
  }
}

object Timer {
  var timerMap = Map[String, Timer]()
  timerMap += ("preprocess" -> new Timer())
  timerMap += ("batch" -> new Timer())
  timerMap += ("inference" -> new Timer())
  timerMap += ("postprocess" -> new Timer())
  timerMap += ("other" -> new Timer())

  def timing[T](name: String, num: Int)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    Logger.getLogger(getClass).info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    if (timerMap.contains(name)) {
      updateTimer(timerMap(name), cost, num)
    } else {
      updateTimer(timerMap("others"), cost, num)
    }
    result
  }
  def updateTimer(timer: Timer, cost: Long, num: Int): Unit = {
    timer.total += cost.toFloat
    timer.record += num
    timer.batchNum += 1
    timer.average = timer.total / timer.record
    timer.averageBatch = timer.total / timer.batchNum
    if (timer.max == Long.MinValue) {
      timer.first = cost
    }
    if (cost > timer.max) {
      timer.max = cost
    }
    if (cost < timer.min) {
      timer.min = cost
    }
    if (timer.topQ.size() >= timer.nQ && cost > timer.topQ.peek()) {
      timer.topQ.poll()
      timer.topQ.add(cost)
    }
    if (timer.topQ.size() < timer.nQ) {
      timer.topQ.add(cost)
    }
  }
  def print(): Unit = {
    var totalTime: Float = 0
    timerMap.foreach(kv => {
      println(s"Time stat for ${kv._1} up to now")
      kv._2.print()
      totalTime += kv._2.total
    })
    println(s"Total time of statistic $totalTime ms")
  }
}

/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.serving.engine

import java.util.PriorityQueue

import org.apache.log4j.Logger

/**
 * Timer class
 * @param _countFlag To determine whether this timer should be count into total
 *                   Sequential operations in main workflow should be count
 *                   but parallel operations in one stage should not
 *                   e.g. The whole cost of preprocessing should be taken,
 *                   but the preprocessing time per image, which is counted in
 *                   parallel manner, should be set false, otherwise the statistic
 *                   count time would be wrong
 */
class Timer(_countFlag: Boolean = true) {
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
  val countFlag = _countFlag
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

/**
 * Singleton class Timer
 * Cluster Serving main workflow contains 4 steps,
 * preprocess, batch, inference, postprocess
 * these 4 steps are taken into total statistics
 * In addition to these, developers could add Timer anywhere in code
 * and the statistics would not be taken into total workflow timing statistics
 * To timing a piece of code, just use
 * Timer.timing(snippet_name, record_number_in_this_snippet)(code_snippet)
 */
object Timer {
  var timerMap = Map[String, Timer]()
  timerMap += ("preprocess" -> new Timer())
  timerMap += ("batch" -> new Timer())
  timerMap += ("inference" -> new Timer())
  timerMap += ("postprocess" -> new Timer())

  def timing[T](name: String, num: Int)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    Logger.getLogger(getClass).info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    if (!timerMap.contains(name)) {
      timerMap += (name -> new Timer(false))
    }
    updateTimer(timerMap(name), cost, num)
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
      if (kv._2.countFlag) {
        totalTime += kv._2.total
      }
    })
    println(s"Total time of statistic $totalTime ms")
    timerMap.foreach(kv => {
      if (kv._2.countFlag) {
        println(s"${kv._1} time cost total percentage ${(kv._2.total/totalTime) * 100} %")
      }
    })
  }
}

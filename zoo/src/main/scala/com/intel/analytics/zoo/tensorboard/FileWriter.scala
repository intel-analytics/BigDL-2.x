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

package com.intel.analytics.zoo.tensorboard

import com.intel.analytics.bigdl.utils.{Engine, ThreadPool}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.tensorflow
import org.tensorflow.framework.GraphDef
import org.tensorflow.util.Event

/**
 * Writes Summary protocol buffers to event files.
 * @param logDirectory Support local directory and HDFS directory
 * @param flushMillis Interval to flush events queue.
 */
private[zoo] class FileWriter(val logDirectory : String, flushMillis: Int = 1000) {
  private val logPath = new Path(logDirectory)
  // write to local disk by default
  private val fs = logPath.getFileSystem(new Configuration(false))

  require(!fs.exists(logPath) || fs.isDirectory(logPath), s"FileWriter: can not create $logPath")
  if (!fs.exists(logPath)) fs.mkdirs(logPath)

  private val eventWriter = new EventWriter(logDirectory, flushMillis, fs)
  val engineObj = "com.intel.analytics.bigdl.utils.Engine"
  val engineDefault = Engine.getClass().getMethod("default")
  val threadPool = engineDefault.invoke(engineObj).asInstanceOf[ThreadPool]
  threadPool.invoke(() => eventWriter.run())

  /**
   * Adds a Summary protocol buffer to the event file.
   * @param summary a Summary protobuf String generated by bigdl.utils.Summary's
   *                scalar()/histogram().
   * @param globalStep a consistent global count of the event.
   * @return
   */
  def addSummary(summary: tensorflow.framework.Summary, globalStep: Long): this.type = {
    val event = Event.newBuilder().setSummary(summary).build()
    addEvent(event, globalStep)
    this
  }

  def addGraphDef(graph: GraphDef): this.type = {
    val event = Event.newBuilder().setGraphDef(graph.toByteString).build()
    eventWriter.addEvent(event)
    this
  }

  /**
   * Add a event protocol buffer to the event file.
   * @param event A event protobuf contains summary protobuf.
   * @param globalStep a consistent global count of the event.
   * @return
   */
  def addEvent(event: Event, globalStep: Long): this.type = {
    eventWriter.addEvent(
      event.toBuilder.setWallTime(System.currentTimeMillis() / 1e3).setStep(globalStep).build())
    this
  }

  /**
   * Close file writer.
   * @return
   */
  def close(): Unit = {
    eventWriter.close()
    fs.close()
  }
}


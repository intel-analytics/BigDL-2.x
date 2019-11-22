package com.intel.analytics.zoo.tensorboard

import java.net.InetAddress
import java.util.concurrent.{LinkedBlockingDeque, TimeUnit}

import org.apache.hadoop.fs.{FileSystem, Path}
import org.tensorflow.util.Event

/**
 * Event writer, write event protocol buffers to file.
 *
 * @param logDir Support local directory and HDFS directory
 * @param flushMillis
 */
private[zoo] class EventWriter(logDir: String,
                                 flushMillis: Int = 1000,
                                 fs: FileSystem) extends Runnable {
  private val eventQueue = new LinkedBlockingDeque[Event]()
  private val outputFile = new Path(logDir +
    s"/bigdl.tfevents.${(System.currentTimeMillis() / 1e3).toInt}" +
    s".${InetAddress.getLocalHost().getHostName()}")
  private val recordWriter = new RecordWriter(outputFile, fs)
  // Add an empty Event to the queue.
  eventQueue.add(Event.newBuilder().setWallTime(System.currentTimeMillis() / 1e3).build())
  @volatile private var running: Boolean = true

  def addEvent(event: Event): this.type = {
    eventQueue.add(event)
    this
  }

  private def flush(): this.type = {
    while (!eventQueue.isEmpty) {
      recordWriter.write(eventQueue.pop())
    }
    this
  }

  private def writeEvent(): this.type = {
    val e = eventQueue.poll(flushMillis, TimeUnit.MILLISECONDS)
    if (null != e) recordWriter.write(e)
    this
  }

  def close(): this.type = {
    running = false
    this
  }

  override def run(): Unit = {
    while (running) {
      writeEvent()
    }
    flush()
    recordWriter.close()
  }
}


package com.intel.analytics.zoo.serving.spark

import java.util.AbstractMap.SimpleEntry

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.receiver.Receiver
import redis.clients.jedis.{Jedis, StreamEntryID}

import scala.collection.JavaConversions._


class ServingReceiver ()
  extends Receiver[(String, String)](StorageLevel.MEMORY_ONLY){

  override def onStart(): Unit = {
    val jedis = new Jedis("localhost", 6379)
    try {
      jedis.xgroupCreate("image_stream", "serving",
        new StreamEntryID(0, 0), true)
    } catch {
      case e: Exception =>
        println(s"$e exist group")
    }
    jedis.xreadGroup(
      "serving",
      "cli",
      512,
      50,
      false,
       new SimpleEntry("image_stream", new StreamEntryID(0, 0)))
    while (!isStopped) {
      val response = jedis.xreadGroup(
        "serving",
        "cli",
        512,
        50,
        false,
        new SimpleEntry("image_stream", StreamEntryID.UNRECEIVED_ENTRY)
      )

      if (response != null) {
        for (streamMessages <- response) {
          println(s"receiving!!! ${streamMessages.getValue.size()}")
          val key = streamMessages.getKey
          val entries = streamMessages.getValue
          val it = entries.map { e =>
            (e.getFields.get("uri"), e.getFields.get("image"))
          }
          store(it.iterator)
//          var i = 0
//          for (e <- entries) {
//            var p = jedis.pipelined()
//              p.xack("image_stream", "serving", e.getID)
//            i = i + 1
//            if (i % 1 == 0) {
//              p.sync()
//              p = jedis.pipelined()
//            }
//          }
        }
      }
    }

  }

  override def onStop(): Unit = {

  }
}

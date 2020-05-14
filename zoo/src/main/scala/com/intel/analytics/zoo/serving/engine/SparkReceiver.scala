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

import java.util.AbstractMap.SimpleEntry

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.receiver.Receiver
import redis.clients.jedis.{Jedis, StreamEntryID}

import scala.collection.JavaConverters._


class ServingReceiver ()
  extends Receiver[(String, String)](StorageLevel.MEMORY_ONLY) {

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
      ).asScala
      Thread.sleep(10)
      if (response != null) {
        for (streamMessages <- response) {
          //          println(s"receiving!!! ${streamMessages.getValue.size()}")
          val key = streamMessages.getKey
          val entries = streamMessages.getValue.asScala
          //          val it = entries.map { e =>
          //            (e.getFields.get("uri"), e.getFields.get("image"))
          //          }.toIterator
          //          store(it)
          val ppl = jedis.pipelined()
          entries.foreach(e => {
            val d = (e.getFields.get("uri"), e.getFields.get("image"))
            store(d)
            ppl.xack("image_stream", "serving", e.getID)
          })
          ppl.sync()
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


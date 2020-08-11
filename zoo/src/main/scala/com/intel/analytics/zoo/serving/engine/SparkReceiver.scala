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

import com.intel.analytics.zoo.serving.utils.Conventions
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.receiver.Receiver
import redis.clients.jedis.{Jedis, StreamEntryID}

import scala.collection.JavaConverters._


class ServingReceiver (redisHost: String = "localhost", redisPort: Int = 6379)
  extends Receiver[(String, String)](StorageLevel.MEMORY_ONLY) {

  override def onStart(): Unit = {
    val jedis = new Jedis(redisHost, redisPort)
    try {
      jedis.xgroupCreate(Conventions.SERVING_STREAM_NAME, "serving",
        new StreamEntryID(0, 0), true)
    } catch {
      case e: Exception =>
        println(s"$e exist group")
    }
//    jedis.xreadGroup(
//      "serving",
//      "cli",
//      64,
//      1,
//      false,
//      new SimpleEntry(Conventions.SERVING_STREAM_NAME, new StreamEntryID(0, 0)))
    while (!isStopped) {
      val response = jedis.xreadGroup(
        "serving",
        "cli",
        64,
        1,
        false,
        new SimpleEntry(Conventions.SERVING_STREAM_NAME, StreamEntryID.UNRECEIVED_ENTRY)
      )
      Thread.sleep(10)
      if (response != null) {
        for (streamMessages <- response.asScala) {
          //          println(s"receiving!!! ${streamMessages.getValue.size()}")
          val key = streamMessages.getKey
          val entries = streamMessages.getValue.asScala
          val ppl = jedis.pipelined()
          entries.foreach(e => {
            val d = (e.getFields.get("uri"), e.getFields.get("data"))
            store(d)
            ppl.xack(Conventions.SERVING_STREAM_NAME, "serving", e.getID)
          })
          ppl.sync()
        }
      }
    }

  }

  override def onStop(): Unit = {

  }
}


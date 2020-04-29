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

import com.intel.analytics.zoo.serving.pipeline.{RedisIO, RedisUtils}
import com.intel.analytics.zoo.serving.utils.{FileUtils, SerParams}
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.source.{RichSourceFunction, SourceFunction}
import redis.clients.jedis.{Jedis, JedisPool, StreamEntryID}

import scala.collection.JavaConverters._

class FlinkRedisSource(params: SerParams) extends RichSourceFunction[List[(String, String)]] {
  @volatile var isRunning = true

  var redisPool: JedisPool = null
  var db: Jedis = null


  override def open(parameters: Configuration): Unit = {
    redisPool = new JedisPool(params.redisHost, params.redisPort)
    db = RedisIO.getRedisClient(redisPool)
    try {
      db.xgroupCreate("image_stream", "serving",
        new StreamEntryID(0, 0), true)
    } catch {
      case e: Exception =>
        println(s"$e exist group")
    }

  }

  override def run(sourceContext: SourceFunction
    .SourceContext[List[(String, String)]]): Unit = while (isRunning) {
    val response = db.xreadGroup(
      "serving",
      "cli",
      512,
      50,
      false,
      new SimpleEntry("image_stream", StreamEntryID.UNRECEIVED_ENTRY))
    if (response != null) {
      for (streamMessages <- response.asScala) {
        val key = streamMessages.getKey
        val entries = streamMessages.getValue.asScala
        val it = entries.map(e => {
          (e.getFields.get("uri"), e.getFields.get("image"))
        }).toList
        sourceContext.collect(it)
      }
      RedisUtils.checkMemory(db, 0.6, 0.5)
      Thread.sleep(10)
    }
    if (FileUtils.checkStop()) {
      isRunning = false
    }
  }

  override def cancel(): Unit = {
    redisPool.close()
  }

}

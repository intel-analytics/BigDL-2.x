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

package com.intel.analytics.zoo.serving

import java.util
import java.util.{AbstractMap, Base64, UUID}

import com.intel.analytics.zoo.serving.http._
import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig, StreamEntryID}

import scala.collection.JavaConverters._

object MockClusterServing extends App with Supportive {
  val redisHost = "localhost"
  val redisPort = 6379

  val jedisPoolConfig = new JedisPoolConfig()
  val jedisPool = new JedisPool(new JedisPoolConfig(), redisHost, redisPort)
  val jedis = jedisPool.getResource()

  val redisInputQueue = "serving_stream"
  val redisOutputQueue = "result:"

  var redisStreamBatchCount = 56
  var redisStreamBlockMillis = 1
  val stream = new AbstractMap.SimpleImmutableEntry[String, StreamEntryID](
    redisInputQueue, StreamEntryID.UNRECEIVED_ENTRY)
  val groupName = "consumer_group"
  val consumerName = s"consumer_${UUID.randomUUID().toString}"

  timing("flushall")() {
    jedis.flushAll()
  }

  timing(s"$this create redis consumer group")() {
    createConsumerGroupIfNotExist(jedis, redisInputQueue, groupName, StreamEntryID.LAST_ENTRY)
  }

  def createConsumerGroupIfNotExist(conn: Jedis,
      streamKey: String, groupName: String, offset: StreamEntryID): Unit = {
    try {
      conn.xgroupCreate(streamKey, groupName, offset, true)
    } catch {
      case e: Exception if e.getMessage.contains("already exists") =>
        logger.info(s"Consumer group already exists: $groupName")
    }
  }

  def source(): List[PredictionInput] = {
    silent(s"$this get result from redis")()  {
      val range = silent(s"$this xread from redis")() {
        jedis.xreadGroup(groupName, consumerName,
          redisStreamBatchCount, redisStreamBlockMillis, true, stream)
      }
      if(null != range && ! range.isEmpty) {
        val items = timing(s"$this parse the range")() {
          val entries = range.get(0).getValue
          val size = entries.size()
          logger.info(s"$this [GET] $size result from redis")
          entries.asScala.toList.map(entry => {
            val id = entry.getID
            val fields = entry.getFields
            val data = fields.get("data")
            val bytes = Base64.getDecoder.decode(data)
            val instances = Instances.fromArrow(bytes)
            val uri = fields.get("uri")
            val item = InstancesPredictionInput(uri, instances)
            item
          })
        }
        println(s"${items.size} items read from redis")
        println(s"${items} read from redis")
        items
      }
      else {
        List()
      }
    }
  }

  def sink(items: List[PredictionInput]): Unit = {
    if (items.size != 0) {
      timing(s"$this [PUT] ${items.size} results to redis")() {
        val pipeline = jedis.pipelined()
        if(items.size < 10) {
          Thread.sleep(50)
        } else {
          val time = (items.size / 10) * 100
          Thread.sleep(time)
        }
        items.map(item => {
          val key = s"${redisOutputQueue}${item.getId()}"
          val value = new util.HashMap[String, String]()
          value.put("result", "mock-result")
          println(key, value)
          pipeline.hmset(key, value)
        })
        pipeline.sync()
      }
    }
  }

  while(true) {
    sink(source())
  }

}

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

package com.intel.analytics.zoo.serving.pipeline

import org.apache.log4j.Logger
import redis.clients.jedis.exceptions.JedisConnectionException
import redis.clients.jedis.{Jedis, JedisPool, Pipeline}

import scala.collection.JavaConverters._


object RedisIO {
  val logger = Logger.getLogger(getClass)
  def getRedisClient(redisPool: JedisPool): Jedis = {
    var jedis: Jedis = null
    var cnt: Int = 0
    while (jedis == null) {
      try {
        jedis = redisPool.getResource
      }
      catch {
        case e: JedisConnectionException =>
          logger.info(s"Redis client can not connect, maybe max number of clients is reached." +
            "Waiting, if you always receive this, please stop your service and report bug.")
          e.printStackTrace()
          cnt += 1
          if (cnt >= 10) {
            throw new Error("can not get redis from the pool")
          }
          Thread.sleep(500)
      }
      Thread.sleep(10)
    }
    return jedis
  }
  def writeHashMap(ppl: Pipeline, key: String, value: String): Unit = {
    val hKey = "result:" + key
    val hValue = Map[String, String]("value" -> value).asJava
    ppl.hmset(hKey, hValue)
  }
}

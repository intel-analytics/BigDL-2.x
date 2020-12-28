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

import com.intel.analytics.zoo.serving.ClusterServing
import com.intel.analytics.zoo.serving.utils.Conventions
import org.apache.log4j.Logger
import redis.clients.jedis.Jedis

object RedisUtils {
  def checkMemory(db: Jedis, inputThreshold: Double, cutRatio: Double): Unit = {
    var redisInfo = RedisUtils.getMapFromInfo(db.info())
    if (redisInfo("used_memory").toLong >=
      redisInfo("maxmemory").toLong * inputThreshold) {
      ClusterServing.synchronized {
        redisInfo = RedisUtils.getMapFromInfo(db.info())
        if (redisInfo("used_memory").toLong >=
          redisInfo("maxmemory").toLong * inputThreshold) {
          ClusterServing.logger.info(s"Used memory ${redisInfo("used_memory")}, " +
            s"Max memory ${redisInfo("maxmemory")}. Your input data length is " +
            s"${db.xlen(Conventions.SERVING_STREAM_DEFAULT_NAME)}. Removing old data...")
          db.xtrim(Conventions.SERVING_STREAM_DEFAULT_NAME,
            (db.xlen(Conventions.SERVING_STREAM_DEFAULT_NAME) * cutRatio).toLong, true)
          ClusterServing.logger.info(s"Trimmed stream, now your serving stream length is " +
            s"${db.xlen(Conventions.SERVING_STREAM_DEFAULT_NAME)}")
          var cuttedRedisInfo = RedisUtils.getMapFromInfo(db.info())
          while (cuttedRedisInfo("used_memory").toLong >=
            cuttedRedisInfo("maxmemory").toLong * inputThreshold) {
            ClusterServing.logger.info(s"Used memory ${redisInfo("used_memory")}, " +
              s"Max memory ${redisInfo("maxmemory")}. " +
              s"Your result field has exceeded the limit, please dequeue. Will retry in 10 sec..")
            cuttedRedisInfo = RedisUtils.getMapFromInfo(db.info())
            Thread.sleep(10000)
          }
        }
      }
    }
  }


  def getMapFromInfo(info: String): Map[String, String] = {
    var infoMap = Map[String, String]()
    val tabs = info.split("#")

    for (tab <- tabs) {
      if (tab.length > 0) {
        val keys = tab.split("\r\n")

        for (key <- keys) {
          if (key.split(":").size == 2) {
            infoMap += (key.split(":").head ->
              key.split(":").last)
          }
        }
      }
    }

    return infoMap
  }
}

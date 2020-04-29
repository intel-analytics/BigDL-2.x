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

import redis.clients.jedis.{Jedis, JedisPool}

object RedisPool {
  def getRedisConnection(pool: JedisPool): Jedis = {
    var connection: Jedis = null
    while (connection == null) {
      Thread.sleep(10)
      try {
        connection = pool.getResource
      }
      catch {
        case e: Exception => throw e
      }
    }
    connection
  }
}

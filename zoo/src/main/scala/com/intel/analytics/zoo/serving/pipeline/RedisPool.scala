package com.intel.analytics.zoo.serving.pipeline

import redis.clients.jedis.{Jedis, JedisPool}

object RedisPool {
  def getRedisConnection(pool: JedisPool): Jedis = {
    var connection: Jedis = null
    while (connection == null) {
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

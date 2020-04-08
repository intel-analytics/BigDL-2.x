package com.intel.analytics.zoo.serving.pipeline

import org.apache.log4j.Logger
import redis.clients.jedis.Jedis

object RedisUtils {
  def checkMemory(db: Jedis, inputThreshold: Double, cutRatio: Double): Unit = {
    val redisInfo = RedisUtils.getMapFromInfo(db.info())
    val logger = Logger.getLogger(getClass)
    if (redisInfo("used_memory").toLong >=
      redisInfo("maxmemory").toLong * inputThreshold) {
      logger.info(s"Used memory ${redisInfo("used_memory")}, " +
        s"Max memory ${redisInfo("maxmemory")}. Trimming old redis stream...")
      db.xtrim("image_stream",
        (db.xlen("image_stream") * cutRatio).toLong, true)
      val cuttedRedisInfo = RedisUtils.getMapFromInfo(db.info())
      while (cuttedRedisInfo("used_memory").toLong >=
        cuttedRedisInfo("maxmemory").toLong * inputThreshold) {
        logger.info(s"Used memory ${redisInfo("used_memory")}, " +
          s"Max memory ${redisInfo("maxmemory")}. Your result field has exceeded the limit, please dequeue.")
        Thread.sleep(10000)
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

package com.intel.analytics.zoo.serving.engine

import java.util.AbstractMap.SimpleEntry

import com.intel.analytics.zoo.serving.pipeline.{RedisIO, RedisUtils}
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.source.{RichSourceFunction, SourceFunction}
import org.apache.log4j.Logger
import redis.clients.jedis.{Jedis, JedisPool, StreamEntryID}
import com.intel.analytics.zoo.serving.ClusterServing.params

import scala.collection.JavaConversions._

class FlinkRedisSource() extends RichSourceFunction[List[(String, String)]] {
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

  override def run(sourceContext: SourceFunction.SourceContext[List[(String,String)]]): Unit =

    while (isRunning){
    val response = db.xreadGroup(
      "serving",
      "cli",
      512,
      50,
      false,
      new SimpleEntry("image_stream", StreamEntryID.UNRECEIVED_ENTRY))
    if (response != null) {
      for (streamMessages <- response) {
        val key = streamMessages.getKey
        val entries = streamMessages.getValue
        val it = entries.map(e => {
          (e.getFields.get("uri"), e.getFields.get("image"))
        }).toList
        sourceContext.collect(it)
      }
      RedisUtils.checkMemory(db, 0.6, 0.5)
      Thread.sleep(10)
    }
  }

  override def cancel(): Unit = {
    db.close()
  }

}

package com.intel.analytics.zoo.serving.engine

import com.intel.analytics.zoo.serving.pipeline.RedisIO
import com.intel.analytics.zoo.serving.utils.SerParams
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.sink.{RichSinkFunction, SinkFunction}
import redis.clients.jedis.{Jedis, JedisPool}
import com.intel.analytics.zoo.serving.ClusterServing.params


class FlinkRedisSink() extends RichSinkFunction[List[(String, String)]] {
  var redisPool: JedisPool = null
  var db: Jedis = null

  override def open(parameters: Configuration): Unit = {
    redisPool = new JedisPool(params.redisHost, params.redisPort)
  }

  override def close(): Unit = {
    redisPool.close()
  }

  override def invoke(value: List[(String, String)]): Unit = {
    params.logger.info(s"Preparing to write result to redis")
    db = RedisIO.getRedisClient(redisPool)
    val ppl = db.pipelined()
    value.foreach(v => RedisIO.writeHashMap(ppl, v._1, v._2))
    ppl.sync()
    db.close()
    params.logger.info(s"${value.size} records written to redis")
  }

}

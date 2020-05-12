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

import com.intel.analytics.zoo.serving.pipeline.RedisIO
import com.intel.analytics.zoo.serving.utils.SerParams
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.sink.{RichSinkFunction, SinkFunction}
import org.apache.log4j.Logger
import redis.clients.jedis.{Jedis, JedisPool}


class FlinkRedisSink(params: SerParams) extends RichSinkFunction[List[(String, String)]] {
  var redisPool: JedisPool = null
  var jedis: Jedis = null
  var logger: Logger = null
  override def open(parameters: Configuration): Unit = {
    redisPool = new JedisPool(params.redisHost, params.redisPort)
//    db = RedisIO.getRedisClient(redisPool)
    logger = Logger.getLogger(getClass)
  }

  override def close(): Unit = {
    redisPool.close()
  }


  override def invoke(value: List[(String, String)]): Unit = {
//    logger.info(s"Preparing to write result to redis")
    jedis = RedisIO.getRedisClient(redisPool)
    val ppl = jedis.pipelined()
    value.foreach(v => RedisIO.writeHashMap(ppl, v._1, v._2))
    ppl.sync()
    jedis.close()
    logger.info(s"${value.size} records written to redis")

  }

}

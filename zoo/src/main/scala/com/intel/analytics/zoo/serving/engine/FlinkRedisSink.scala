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
import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig}


class FlinkRedisSink(params: SerParams) extends RichSinkFunction[List[(String, String)]] {
  var redisPool: JedisPool = null
  var jedis: Jedis = null
  var logger: Logger = null

  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)

    if (params.redisSecureEnabled) {
      System.setProperty("javax.net.ssl.trustStore", params.redisSecureTrustStorePath)
      System.setProperty("javax.net.ssl.trustStorePassword", params.redisSecureTrustStorePassword)
      System.setProperty("javax.net.ssl.keyStoreType", "JKS")
      System.setProperty("javax.net.ssl.keyStore", params.redisSecureTrustStorePath)
      System.setProperty("javax.net.ssl.keyStorePassword", params.redisSecureTrustStorePassword)
    }

    redisPool = new JedisPool(new JedisPoolConfig(),
      params.redisHost, params.redisPort, params.redisSecureEnabled)
    params.redisSecureEnabled match {
      case true => logger.info(s"FlinkRedisSink connect to secured Redis successfully.")
      case false => logger.info(s"FlinkRedisSink connect to plain Redis successfully.")
    }
    jedis = RedisIO.getRedisClient(redisPool)

  }

  override def close(): Unit = {
    if (null != redisPool) {
      redisPool.close()
    }
  }


  override def invoke(value: List[(String, String)]): Unit = {
//    logger.info(s"Preparing to write result to redis")

    val ppl = jedis.pipelined()
    value.foreach(v => RedisIO.writeHashMap(ppl, v._1, v._2))
    ppl.sync()
    jedis.close()
    logger.info(s"${value.size} records written to redis")

  }

}

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

import java.util.AbstractMap.SimpleEntry
import java.util.UUID

import com.intel.analytics.zoo.serving.pipeline.{RedisIO, RedisUtils}
import com.intel.analytics.zoo.serving.utils.{Conventions, FileUtils, SerParams}
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.source.{RichParallelSourceFunction, RichSourceFunction, SourceFunction}
import org.apache.log4j.Logger
import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig, StreamEntryID}

import scala.collection.JavaConverters._

class FlinkRedisSource(params: SerParams)
  extends RichParallelSourceFunction[List[(String, String)]] {
  @volatile var isRunning = true

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
      params.redisHost, params.redisPort, 5000, params.redisSecureEnabled)
    params.redisSecureEnabled match {
      case true => logger.info(s"FlinkRedisSource connect to secured Redis successfully.")
      case false => logger.info(s"FlinkRedisSource connect to plain Redis successfully.")
    }
    jedis = RedisIO.getRedisClient(redisPool)
    try {
      jedis.xgroupCreate(Conventions.SERVING_STREAM_NAME, "serving", new StreamEntryID(0, 0), true)
    } catch {
      case e: Exception =>
        println(s"$e exist group")
    }
  }

  override def run(sourceContext: SourceFunction
    .SourceContext[List[(String, String)]]): Unit = while (isRunning) {
    // logger.info(s">>> get from source begin ${System.currentTimeMillis()} ms")
    val start = System.nanoTime()
    val groupName = "serving"
    val consumerName = "consumer-" + UUID.randomUUID().toString
    val response = jedis.xreadGroup(
      groupName,
      consumerName,
      params.coreNum,
      1,
      false,
      new SimpleEntry(Conventions.SERVING_STREAM_NAME, StreamEntryID.UNRECEIVED_ENTRY))
    // logger.info(s">>> get from source readed redis ${System.currentTimeMillis()} ms")
    if (response != null) {
      for (streamMessages <- response.asScala) {
        val key = streamMessages.getKey
        val entries = streamMessages.getValue.asScala
        val it = entries.map(e => {
          (e.getFields.get("uri"), e.getFields.get("data"))
        }).toList
        sourceContext.collect(it)
      }
      RedisUtils.checkMemory(jedis, 0.6, 0.5)
    }
  }

  override def cancel(): Unit = {
    jedis.close()
    redisPool.close()
    logger.info("Flink source cancelled")
  }

}

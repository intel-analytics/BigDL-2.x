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

import com.intel.analytics.zoo.serving.ClusterServing
import com.intel.analytics.zoo.serving.pipeline.RedisUtils
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, Conventions}
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.sink.{RichSinkFunction, SinkFunction}
import org.apache.log4j.Logger
import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig, StreamEntryID}



class FlinkRedisSink(helperSer: ClusterServingHelper)
  extends RichSinkFunction[List[(String, String)]] {
  var jedis: Jedis = null
  var logger: Logger = null
  var helper: ClusterServingHelper = null
  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)
    // Sink is first initialized among Source, Map, Sink, so initialize static variable in sink.
    ClusterServing.helper = helperSer
    helper = ClusterServing.helper
    RedisUtils.initializeRedis()
    jedis = RedisUtils.getRedisClient(ClusterServing.jedisPool)
  }

  override def close(): Unit = {
    if (null != jedis) {
      jedis.close()
    }
  }

  override def invoke(value: List[(String, String)], context: SinkFunction.Context[_]): Unit = {
    val ppl = jedis.pipelined()
    var cnt = 0
    value.foreach(v => {
      RedisUtils.writeHashMap(ppl, v._1, v._2, helper.jobName)
      if (v._2 != "NaN") {
        cnt += 1
      }
    })
    ppl.sync()
    logger.info(s"${cnt} valid records written to redis")
  }

}


class FlinkRedisXStreamSink(helper: ClusterServingHelper) extends FlinkRedisSink(helper) {
  override def invoke(value: List[(String, String)], context: SinkFunction.Context[_]): Unit = {
    val ppl = jedis.pipelined()
    var cnt = 0
    value.foreach(v => {
      RedisUtils.writeXstream(ppl, v._1, v._2, helper.jobName)
      if (v._2 != "NaN") {
        cnt += 1
      }
    })
    ppl.sync()
    logger.info(s"${cnt} valid records written to redis")
  }
}


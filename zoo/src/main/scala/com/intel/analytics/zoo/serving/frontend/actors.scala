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

package com.intel.analytics.zoo.serving.frontend

import java.util
import java.util.Map
import java.util.concurrent.TimeUnit

import akka.actor.{Actor, ActorRef}
import org.slf4j.LoggerFactory
import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig}

import scala.collection.mutable.{Set => MutableSet}
import scala.concurrent.Await
import akka.pattern.ask
import akka.util.Timeout

trait JedisEnabledActor extends Actor with Supportive {
  val actorName = self.path.name

  def retrieveJedis(redisHost: String, redisPort: Int): Jedis =
    timing(s"$actorName retrieve redis connection")() {
    val jedisPoolConfig = new JedisPoolConfig()
    val jedisPool = new JedisPool(new JedisPoolConfig(), redisHost, redisPort)
    jedisPool.getResource()
  }
}


class RedisPutActor(
    redisHost: String,
    redisPort: Int,
    redisInputQueue: String,
    redisOutputQueue: String,
    timeWindow: Int,
    countWindow: Int) extends JedisEnabledActor {
  override val logger = LoggerFactory.getLogger(classOf[RedisPutActor])
  val jedis = retrieveJedis(redisHost, redisPort)

  var start = System.currentTimeMillis()
  val set = MutableSet[PredictionInput]()

  override def receive: Receive = {
    case message: PredictionInputMessage =>
      silent(s"$actorName input message process, ${set.size}")() {
        val predictionInput = message.input
        set.add(predictionInput)
      }
    case mesage: PredictionInputFlushMessage =>
      silent(s"$actorName flush message process, ${set.size}")() {
        val now = System.currentTimeMillis()
        val interval = now - start
        val setSize = set.size
        if (setSize != 0) {
          logger.info(s"$actorName flush inpus with $interval, $setSize")
          if (interval >= timeWindow || setSize >= countWindow) {
            silent(s"$actorName put message process")() {
              putInTransaction(redisInputQueue, set)
            }
            start = System.currentTimeMillis()
          }
        }
      }
  }

  def put(queue: String, input: PredictionInput): Unit = {
    timing(s"$actorName put request to redis")(FrontEndApp.putRedisTimer) {
      val hash = input.toHash()
      jedis.xadd(queue, null, hash)
    }
  }

  def putInPipeline(queue: String, inputs: MutableSet[PredictionInput]): Unit = {
    average(s"$actorName put ${inputs.size} requests to redis")(inputs.size)(
      FrontEndApp.putRedisTimer) {
      val pipeline = jedis.pipelined()
      inputs.map(input => {
        val hash = input.toHash()
        pipeline.xadd(queue, null, hash)
      })
      pipeline.sync()
      inputs.clear()
    }
  }

  def putInTransaction(queue: String, inputs: MutableSet[PredictionInput]): Unit = {
    average(s"$actorName put ${inputs.size} requests to redis")(inputs.size)(
      FrontEndApp.putRedisTimer) {
      val t = jedis.multi();
      inputs.map(input => {
        val hash = input.toHash()
        t.xadd(queue, null, hash)
      })
      t.exec()
      logger.info(s"${System.currentTimeMillis}, ${inputs.map(_.getId).mkString(",")}")
      inputs.clear()
    }
  }
}

class RedisGetActor(
    redisHost: String,
    redisPort: Int,
    redisInputQueue: String,
    redisOutputQueue: String) extends JedisEnabledActor {
  override val logger = LoggerFactory.getLogger(classOf[RedisPutActor])
  val jedis = retrieveJedis(redisHost, redisPort)

  override def receive: Receive = {
    case message: PredictionOutputMessage =>
      val result = get(redisOutputQueue, message.id)
      if (null != result && !result.isEmpty) {
        sender() ! result.toString
      } else {
        sender() ! ""
      }
  }

  def get(queue: String, id: String): util.Map[String, String] = {
    silent(s"$actorName get response from redis")(FrontEndApp.getRedisTimer) {
      val key = s"$queue$id"
      jedis.hgetAll(key)
    }
  }
}

class QueryActor(redisGetActor: ActorRef) extends JedisEnabledActor {
  override val logger = LoggerFactory.getLogger(classOf[RedisPutActor])
  val system = context.system
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  override def receive: Receive = {
    case message: PredictionOutputMessage =>
      var result = ""
      timing(s"$actorName waiting")(FrontEndApp.waitRedisTimer) {
        while ("" == result) {
          result = Await.result(redisGetActor ? message, timeout.duration).asInstanceOf[String]
        }
      }
  }
}
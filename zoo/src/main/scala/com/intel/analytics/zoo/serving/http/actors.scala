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

package com.intel.analytics.zoo.serving.http

import java.util
import java.util.concurrent.TimeUnit

import akka.actor.{Actor, ActorRef}
import org.slf4j.LoggerFactory
import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig}

import scala.collection.mutable.{Set => MutableSet}
import scala.concurrent.Await
import scala.concurrent.duration.DurationInt
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
  val cache = MutableSet[PredictionInput]()

  override def receive: Receive = {
    case message: PredictionInputMessage =>
      silent(s"$actorName input message process, ${cache.size}")() {
        val predictionInputs = message.inputs
        predictionInputs.foreach(cache.add(_))
      }
    case mesage: PredictionInputFlushMessage =>
      silent(s"$actorName flush message process, ${cache.size}")() {
        val now = System.currentTimeMillis()
        val interval = now - start
        val setSize = cache.size
        if (setSize != 0) {
          logger.info(s"$actorName flush inpus with interval:$interval, size:$setSize")
          if (interval >= timeWindow || setSize >= countWindow) {
            timing(s"$actorName put message process")() {
              putInTransaction(redisInputQueue, cache)
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
    case message: PredictionQueryMessage =>
      val results = get(redisOutputQueue, message.ids)
      if (null != results && results.size == message.ids.size) {
        sender() ! results
      } else {
        sender() ! Seq[(String, util.Map[String, String])]()
      }
  }

  def get(queue: String, ids: Seq[String]): Seq[(String, util.Map[String, String])] = {
    silent(s"$actorName get response from redis")(FrontEndApp.getRedisTimer) {
      ids.map(id => {
        val key = s"$queue$id"
        (id, jedis.hgetAll(key))
      }).filter(!_._2.isEmpty)
    }
  }
}

class QueryActor(redisGetActor: ActorRef) extends JedisEnabledActor {
  override val logger = LoggerFactory.getLogger(classOf[RedisPutActor])
  val system = context.system
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  override def receive: Receive = {
    case query: PredictionQueryMessage =>
      val target = sender()
      val message = PredictionQueryWithTargetMessage(query, target)
      // context.system.scheduler.scheduleOnce(10 milliseconds, self, message)
      self ! message
    case message: PredictionQueryWithTargetMessage =>
      val results = silent("query to redisGetActor")() {
        Await.result(redisGetActor ? message.query, timeout.duration)
          .asInstanceOf[Seq[(String, util.Map[String, String])]]
      }
      // println(System.currentTimeMillis(), message.query.id, result)
      if(results.size == 0) {
        context.system.scheduler.scheduleOnce(10 milliseconds, self, message)
      } else {
        message.target ! results
      }
  }
}

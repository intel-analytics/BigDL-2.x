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

package com.intel.analytics.zoo.serving

import java.util
import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.pattern.ask
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.github.fppt.jedismock.RedisServer
import com.intel.analytics.zoo.serving.http.{PredictionInputMessage, _}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import redis.clients.jedis.Jedis

import scala.concurrent.{Await, ExecutionContextExecutor}
import scala.util.Random

class FrontendActorsSpec extends FlatSpec with Matchers with BeforeAndAfter with Supportive {
  val random = new Random()
  val redisHost = "localhost"
  val redisPort = random.nextInt(100) + 10000
  var redisServer: RedisServer = _
  var jedis: Jedis = _

  implicit var system: ActorSystem = _
  implicit var materializer: ActorMaterializer = _
  implicit var executionContext: ExecutionContextExecutor = _
  implicit var timeout: Timeout = _

  val redisInputQueue = "image_stream"
  val redisOutputQueue = "result:"

  val input1 = BytesPredictionInput("aW1hZ2UgYnl0ZXM=")
  val input2 = BytesPredictionInput("aW1hZ2UgYnl0ZXM=")
  val input3 = BytesPredictionInput("aW1hZ2UgYnl0ZXM=")
  val inputMessage1 = PredictionInputMessage(input1)
  val inputMessage2 = PredictionInputMessage(input2)
  val inputMessage3 = PredictionInputMessage(input3)
  val flushMessage = PredictionInputFlushMessage()

  before {
    redisServer = RedisServer.newRedisServer(redisPort)
    redisServer.start()

    jedis = new Jedis(redisHost, redisPort)

    system = ActorSystem("zoo-serving-frontend-system")
    materializer = ActorMaterializer()
    executionContext = system.dispatcher
    timeout = Timeout(10, TimeUnit.SECONDS)
  }

  after {
    redisServer.stop()
    system.terminate()
  }

  "redisServer" should "works well" in {
    redisServer shouldNot be (null)
    redisServer.getBindPort should be (redisPort)
    jedis shouldNot be (null)
  }

  "actors" should "works well" in {
    val redisPutterName = s"redis-putter"
    val redisPutter = timing(s"$redisPutterName initialized.")() {
      val redisPutterProps = Props(new RedisPutActor(redisHost, redisPort,
        redisInputQueue, redisOutputQueue, 0, 56))
      system.actorOf(redisPutterProps, name = redisPutterName)
    }
    val redisGetterName = s"redis-getter"
    val redisGetter = timing(s"$redisGetterName initialized.")() {
      val redisGetterProps = Props(new RedisGetActor(redisHost, redisPort,
        redisInputQueue, redisOutputQueue))
      system.actorOf(redisGetterProps, name = redisGetterName)
    }
    val querierNum = 1
    val querierQueue = timing(s"queriers initialized.")() {
      val querierQueue = new LinkedBlockingQueue[ActorRef](querierNum)
      val querierProps = Props(new QueryActor(redisGetter))
      List.range(0, querierNum).map(index => {
        val querierName = s"querier-$index"
        val querier = system.actorOf(querierProps, name = querierName)
        querierQueue.put(querier)
      })
      querierQueue
    }
    redisPutter shouldNot be (null)
    redisGetter shouldNot be (null)
    querierQueue.size() should be (1)
    redisPutter ! inputMessage1
    redisPutter ! inputMessage2
    redisPutter ! inputMessage3
    // redisPutter ! flushMessage

    // mock the cluster serving doing stuff
    mockClusterServing(inputMessage1, inputMessage2, inputMessage3)

    List(inputMessage1, inputMessage2, inputMessage3).foreach(message => {
      val input = message.input
      val key = input.getId()
      val queryMessage = PredictionQueryMessage(key)
      val querier = silent("querier take")() {
        querierQueue.take()
      }
      val result = timing(s"query message wait for key $key")() {
        Await.result(querier ? queryMessage, timeout.duration).asInstanceOf[String]
      }
      silent("querier back")() {
        querierQueue.offer(querier)
      }
      // println(result)
      result should be ("{result=mock-result}")
    })
  }

  def mockClusterServing(messages: PredictionInputMessage*): Any = {
    messages.foreach(message => {
      val item = message.input
      val key = s"${redisOutputQueue}${item.getId()}"
      val value = new util.HashMap[String, String]()
      value.put("result", "mock-result")
      // println(key, value)
      jedis.hmset(key, value)
    })
  }

}

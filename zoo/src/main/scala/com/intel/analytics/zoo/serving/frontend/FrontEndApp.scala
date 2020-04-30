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

import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.http.scaladsl.Http
import akka.http.scaladsl.server.Directives.complete
import akka.http.scaladsl.server.Directives.path
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.codahale.metrics.MetricRegistry
import org.slf4j.LoggerFactory

object FrontEndApp extends Supportive {
  override val logger = LoggerFactory.getLogger(getClass)

  val name = "analytics zoo web serving frontend"

  implicit val system = ActorSystem("zoo-serving-frontend-system")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  val redisInputQueue = "image_stream"
  val redisOutputQueue = "result:"


  def main(args: Array[String]): Unit = {
    timing(s"$name started successfully.")() {
      val arguments = timing("parse arguments")() {
        argumentsParser.parse(args, FrontEndAppArguments()) match {
          case Some(arguments) => logger.info(s"starting with $arguments"); arguments
          case None => argumentsParser.failure("miss args, please see the usage info"); null
        }
      }

      val redisPutterName = s"redis-putter"
      val redisPutter = timing(s"$redisPutterName initialized.")() {
        val redisPutterProps = Props(new RedisPutActor(arguments.redisHost,
          arguments.redisPort, redisInputQueue, redisOutputQueue))
        system.actorOf(redisPutterProps, name = redisPutterName)
      }

      val redisGetterName = s"redis-getter"
      val redisGetter = timing(s"$redisGetterName initialized.")() {
        val redisGetterProps = Props(new RedisGetActor(arguments.redisHost,
          arguments.redisPort, redisInputQueue, redisOutputQueue))
        system.actorOf(redisGetterProps, name = redisGetterName)
      }

      val querierNum = arguments.parallelism
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

      Http().bindAndHandle(route, arguments.interface, arguments.port)

    }
  }

  val metrics = new MetricRegistry
  val overallRequestTimer = metrics.timer("zoo.serving.request.overall")
  val predictRequestTimer = metrics.timer("zoo.serving.request.predict")
  val inferenceRequestTimer = metrics.timer("zoo.serving.request.inference")
  val putRedisTimer = metrics.timer("zoo.serving.redis.put")
  val getRedisTimer = metrics.timer("zoo.serving.redis.get")
  val waitRedisTimer = metrics.timer("zoo.serving.redis.wait")
  val metricsRequestTimer = metrics.timer("zoo.serving.request.metrics")

  val jacksonJsonSerializer = new JacksonJsonSerializer()

  val route = timing("initialize http route")() {
    path("") {
      timing("welcome")(overallRequestTimer) {
        complete("welcome to " + name)
      }
    } ~ (get & path("metrics")) {
      timing("metrics")(overallRequestTimer, metricsRequestTimer) {
        val keys = metrics.getTimers().keySet()
        val servingMetrics = keys.toArray.map(key => {
          val timer = metrics.getTimers().get(key)
          ServingTimerMetrics(key.toString, timer)
        }).toList
        println(servingMetrics)
        complete(jacksonJsonSerializer.serialize(servingMetrics))
      }
    }
  }

  val argumentsParser = new scopt.OptionParser[FrontEndAppArguments]("AZ Serving") {
    head("Analytics Zoo Serving Frontend")
    opt[String]('i', "interface")
      .action((x, c) => c.copy(interface = x))
      .text("network interface of frontend")
    opt[Int]('p', "port")
      .action((x, c) => c.copy(port = x))
      .text("network port of frontend")
    opt[String]('h', "redisHost")
      .action((x, c) => c.copy(redisHost = x))
      .text("host of redis")
    opt[Int]('r', "redisPort")
      .action((x, c) => c.copy(redisPort = x))
      .text("port of redis")
    opt[Int]('s', "parallelism")
      .action((x, c) => c.copy(parallelism = x))
      .text("parallelism of frontend")
    opt[Int]('t', "timeWindow")
      .action((x, c) => c.copy(timeWindow = x))
      .text("timeWindow of frontend")
    opt[Int]('c', "countWindow")
      .action((x, c) => c.copy(countWindow = x))
      .text("countWindow of frontend")
  }
}

case class FrontEndAppArguments(
    interface: String = "0.0.0.0",
    port: Int = 10020,
    redisHost: String = "localhost",
    redisPort: Int = 6379,
    parallelism: Int = 1000,
    timeWindow: Int = 0,
    countWindow: Int = 56
)

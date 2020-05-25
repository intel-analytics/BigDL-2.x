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
import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.MediaTypes
import akka.http.scaladsl.server.Directives.{complete, path, _}
import akka.pattern.ask
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.codahale.metrics.MetricRegistry
import com.google.common.util.concurrent.RateLimiter
import org.slf4j.LoggerFactory

import scala.concurrent.Await
import scala.concurrent.duration.DurationInt

object FrontEndApp extends Supportive {
  override val logger = LoggerFactory.getLogger(getClass)

  val name = "analytics zoo web serving frontend"

  implicit val system = ActorSystem("zoo-serving-frontend-system")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  val redisInputQueue = "serving_stream"
  val redisOutputQueue = "result:"
  logger.debug(s"will connect redis with queues: $redisInputQueue, $redisOutputQueue")

  def main(args: Array[String]): Unit = {
    timing(s"$name started successfully.")() {
      val arguments = timing("parse arguments")() {
        argumentsParser.parse(args, FrontEndAppArguments()) match {
          case Some(arguments) => logger.info(s"starting with $arguments"); arguments
          case None => argumentsParser.failure("miss args, please see the usage info"); null
        }
      }

      val rateLimiter: RateLimiter = arguments.tokenBucketEnabled match {
        case true => RateLimiter.create(arguments.tokensPerSecond)
        case false => null
      }

      val redisPutterName = s"redis-putter"
      val redisPutter = timing(s"$redisPutterName initialized.")() {
        val redisPutterProps = Props(new RedisPutActor(
          arguments.redisHost, arguments.redisPort,
          redisInputQueue, redisOutputQueue,
          arguments.timeWindow, arguments.countWindow))
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

      def processBytesPredictionInput(inputs: Seq[BytesPredictionInput]):
      Seq[PredictionOutput[String]] = {
        silent("put message send")() {
          val message = PredictionInputMessage(inputs)
          redisPutter ! message
        }
        val result = silent("response waiting")() {
          val ids = inputs.map(_.getId())
          val queryMessage = PredictionQueryMessage(ids)
          val querier = silent("querier take")() {
            querierQueue.take()
          }
          val results = timing(s"query message wait for key $ids")() {
            Await.result(querier ? queryMessage, timeout.duration)
              .asInstanceOf[Seq[(String, util.Map[String, String])]]
          }
          silent("querier back")() {
            querierQueue.offer(querier)
          }
          results.map(r => PredictionOutput(r._1, r._2.toString))
        }
        result
      }

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
            complete(jacksonJsonSerializer.serialize(servingMetrics))
          }
        } ~ (post & path("predict") & extract(_.request.entity.contentType) & entity(as[String])) {
          (contentType, content) => {
            val rejected = arguments.tokenBucketEnabled match {
              case true =>
                if (!rateLimiter.tryAcquire(arguments.tokenAcquireTimeout, TimeUnit.MILLISECONDS)) {
                  true
                } else {
                  false
                }
              case false => false
            }
            if (rejected) {
              val error = ServingError("limited")
              complete(500, error.toString)
            } else {
              contentType.mediaType match {
                case MediaTypes.`text/plain` =>
                  timing("predict")(overallRequestTimer, predictRequestTimer) {
                    val input = silent("parse raw")() {
                      BytesPredictionInput(content)
                    }
                    val outputs = processBytesPredictionInput(Seq(input))
                    val result = Predictions(outputs)
                    silent("response complete")() {
                      complete(200, result.toString)
                    }
                  }
                case MediaTypes.`application/json` =>
                  timing("predict")(overallRequestTimer, predictRequestTimer) {
                    val input = timing("parse json")() {
                      val instances = JsonUtil.fromJson(classOf[Instances], content)
                      instances.instances.map(instance => {
                        val image = instance.get("image").get
                          .asInstanceOf[Map[String, String]].get("b64").get
                        BytesPredictionInput(image)
                      })
                    }
                    val outputs = processBytesPredictionInput(input)
                    val result = Predictions(outputs)
                    silent("response complete")() {
                      complete(200, result.toString)
                    }
                  }
                case _ => silent("response complete")() {
                  val error = ServingError("Unsupported Media Type," +
                    " only support text/plain for raw" +
                    " and application/json for json")
                  complete(415, error.toString)
                }
              }
            }
          }
        }
      }


      Http().bindAndHandle(route, arguments.interface, arguments.port)
      system.scheduler.schedule(10 milliseconds, 10 milliseconds,
        redisPutter, PredictionInputFlushMessage())(system.dispatcher)
    }
  }

  val metrics = new MetricRegistry
  val overallRequestTimer = metrics.timer("zoo.serving.request.overall")
  val predictRequestTimer = metrics.timer("zoo.serving.request.predict")
  val putRedisTimer = metrics.timer("zoo.serving.redis.put")
  val getRedisTimer = metrics.timer("zoo.serving.redis.get")
  val waitRedisTimer = metrics.timer("zoo.serving.redis.wait")
  val metricsRequestTimer = metrics.timer("zoo.serving.request.metrics")

  val jacksonJsonSerializer = new JacksonJsonSerializer()

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
    opt[Boolean]('e', "tokenBucketEnabled")
      .action((x, c) => c.copy(tokenBucketEnabled = x))
      .text("Token Bucket Enabled or not")
    opt[Int]('k', "tokensPerSecond")
      .action((x, c) => c.copy(tokensPerSecond = x))
      .text("tokens per second")
    opt[Int]('a', "tokenAcquireTimeout")
      .action((x, c) => c.copy(tokenAcquireTimeout = x))
      .text("token acquire timeout")
  }
}

case class FrontEndAppArguments(
    interface: String = "0.0.0.0",
    port: Int = 10020,
    redisHost: String = "localhost",
    redisPort: Int = 6379,
    parallelism: Int = 1000,
    timeWindow: Int = 0,
    countWindow: Int = 56,
    tokenBucketEnabled: Boolean = false,
    tokensPerSecond: Int = 100,
    tokenAcquireTimeout: Int = 100
)

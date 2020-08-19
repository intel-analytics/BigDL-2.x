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

import java.io.File
import java.security.{KeyStore, SecureRandom}
import java.util
import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}
import javax.net.ssl.{KeyManagerFactory, SSLContext, TrustManagerFactory}

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.http.scaladsl.{ConnectionContext, Http}
import akka.http.scaladsl.server.Directives.{complete, path, _}
import akka.pattern.ask
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.codahale.metrics.MetricRegistry
import com.google.common.util.concurrent.RateLimiter
import com.intel.analytics.zoo.pipeline.inference.EncryptSupportive
import com.intel.analytics.zoo.serving.utils.Conventions
import org.slf4j.LoggerFactory

import scala.concurrent.Await
import scala.concurrent.duration.DurationInt

object FrontEndApp extends Supportive with EncryptSupportive {
  override val logger = LoggerFactory.getLogger(getClass)

  val name = "analytics zoo web serving frontend"

  implicit val system = ActorSystem("zoo-serving-frontend-system")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

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
          arguments.redisInputQueue, arguments.redisOutputQueue,
          arguments.timeWindow, arguments.countWindow,
          arguments.redisSecureEnabled,
          arguments.redissTrustStorePath,
          arguments.redissTrustStorePassword))
        system.actorOf(redisPutterProps, name = redisPutterName)
      }

      val redisGetterName = s"redis-getter"
      val redisGetter = timing(s"$redisGetterName initialized.")() {
        val redisGetterProps = Props(new RedisGetActor(
          arguments.redisHost,
          arguments.redisPort,
          arguments.redisInputQueue,
          arguments.redisOutputQueue,
          arguments.redisSecureEnabled,
          arguments.redissTrustStorePath,
          arguments.redissTrustStorePassword))
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

      def processPredictionInput(inputs: Seq[PredictionInput]):
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
          val results = timing(s"query message wait for key $ids")(
            overallRequestTimer, waitRedisTimer) {
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
        } ~ (post & path("model-secure") &
          extract(_.request.entity.contentType) & entity(as[String])) {
          (contentType, content) => {
            try {
              val secrets = content.split("&")
              val secret = secrets(0).split("=")(1)
              val salt = secrets(1).split("=")(1)
              val message = SecuredModelSecretSaltMessage(secret, salt)
              val result = Await.result(redisPutter ? message, timeout.duration)
                .asInstanceOf[Boolean]
              result match {
                case true => complete("model secured secrect and salt succeed to put in redis")
                case false => complete("model secured secrect and salt failed to put in redis")
              }
            } catch {
              case e: Exception =>
                e.printStackTrace()
                val error = ServingError(e.getMessage + "\n please post a content like " +
                  "secret=xxx&salt=xxxx")
                complete(500, error.toString)
            }

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
              try {
                val result = timing("predict")(overallRequestTimer, predictRequestTimer) {
                  val instances = timing("json deserialization")() {
                    JsonUtil.fromJson(classOf[Instances], content)
                  }
                  val inputs = instances.instances.map(instance => {
                    InstancesPredictionInput(Instances(instance))
                  })
                  val outputs = processPredictionInput(inputs)
                  Predictions(outputs)
                }
                silent("response complete")() {
                  complete(200, result.toString)
                }
              } catch {
                case e =>
                  val message = e.getMessage
                  val exampleJson =
                    """{
  "instances" : [ {
    "intScalar" : 12345,
    "floatScalar" : 3.14159,
    "stringScalar" : "hello, world. hello, zoo.",
    "intTensor" : [ 7756, 9549, 1094, 9808, 4959, 3831, 3926, 6578, 1870, 1741 ],
    "floatTensor" : [ 0.6804766, 0.30136853, 0.17394465, 0.44770062, 0.20275897 ],
    "stringTensor" : [ "come", "on", "united" ],
    "intTensor2" : [ [ 1, 2 ], [ 3, 4 ], [ 5, 6 ] ],
    "floatTensor2" : [ [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ], [ [ 0.2, 0.3 ], [ 0.5, 0.6 ] ] ],
    "stringTensor2" : [ [ [ [ "come", "on", "united" ], [ "come", "on", "united" ] ] ] ],
    "sparseTensor" : {
      "shape" : [ 100, 10000, 10 ],
      "data" : [ 0.2, 0.5, 3.45, 6.78 ],
      "indices" : [ [ 1, 1, 1 ], [ 2, 2, 2 ], [ 3, 3, 3 ], [ 4, 4, 4 ] ]
    },
    "image": "/9j/4AAQSkZJRgABAQEASABIAAD/7RcEUGhvdG9za..."
  } ]
}"""
                  val error = ServingError(s"Wrong content format.\n" +
                    s"Details: ${message}\n" +
                    s"Please refer to examples:\n" +
                    s"$exampleJson\n")
                  complete(400, error.error)
              }
            }
          }
        }
      }
      if (arguments.httpsEnabled) {
        val serverContext = defineServerContext(arguments.httpsKeyStorePassword,
          arguments.httpsKeyStorePath)
        Http().bindAndHandle(route, arguments.interface, port = arguments.securePort,
          connectionContext = serverContext)
        logger.info(s"https started at https://${arguments.interface}:${arguments.securePort}")
      }
      Http().bindAndHandle(route, arguments.interface, arguments.port)
      logger.info(s"http started at http://${arguments.interface}:${arguments.port}")
      system.scheduler.schedule(10 milliseconds, 1 milliseconds,
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
    opt[Int]('s', "securePort")
      .action((x, c) => c.copy(port = x))
      .text("https port of frontend")
    opt[String]('h', "redisHost")
      .action((x, c) => c.copy(redisHost = x))
      .text("host of redis")
    opt[Int]('r', "redisPort")
      .action((x, c) => c.copy(redisPort = x))
      .text("port of redis")
    opt[String]('i', "redisInputQueue")
      .action((x, c) => c.copy(redisInputQueue = x))
      .text("input queue of redis")
    opt[String]('o', "redisOutputQueue")
      .action((x, c) => c.copy(redisOutputQueue = x))
      .text("output queue  of redis")
    opt[Int]('l', "parallelism")
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
    opt[Boolean]('s', "httpsEnabled")
      .action((x, c) => c.copy(httpsEnabled = x))
      .text("https enabled or not")
    opt[String]('p', "httpsKeyStorePath")
      .action((x, c) => c.copy(httpsKeyStorePath = x))
      .text("https keyStore path")
    opt[String]('w', "httpsKeyStorePassword")
      .action((x, c) => c.copy(httpsKeyStorePassword = x))
      .text("https keyStore password")
    opt[Boolean]('s', "redisSecureEnabled")
      .action((x, c) => c.copy(redisSecureEnabled = x))
      .text("redis secure enabled or not")
    opt[Boolean]('s', "httpsEnabled")
      .action((x, c) => c.copy(httpsEnabled = x))
      .text("https enabled or not")
    opt[String]('p', "redissTrustStorePath")
      .action((x, c) => c.copy(redissTrustStorePath = x))
      .text("rediss trustStore path")
    opt[String]('w', "redissTrustStorePassword")
      .action((x, c) => c.copy(redissTrustStorePassword = x))
      .text("rediss trustStore password")
  }

  def defineServerContext(httpsKeyStorePassword: String,
      httpsKeyStorePath: String): ConnectionContext = {
    val password = httpsKeyStorePassword.toCharArray

    val keyStore = KeyStore.getInstance("PKCS12")
    val keystoreInputStream = new File(httpsKeyStorePath).toURI().toURL().openStream()
    require(keystoreInputStream != null, "Keystore required!")
    keyStore.load(keystoreInputStream, password)

    val keyManagerFactory = KeyManagerFactory.getInstance("SunX509")
    keyManagerFactory.init(keyStore, password)

    val trustManagerFactory = TrustManagerFactory.getInstance("SunX509")
    trustManagerFactory.init(keyStore)

    val sslContext = SSLContext.getInstance("TLS")
    sslContext.init(keyManagerFactory.getKeyManagers,
      trustManagerFactory.getTrustManagers, new SecureRandom)

    ConnectionContext.https(sslContext)
  }
}

case class FrontEndAppArguments(
    interface: String = "0.0.0.0",
    port: Int = 10020,
    securePort: Int = 10023,
    redisHost: String = "localhost",
    redisPort: Int = 6379,
    redisInputQueue: String = Conventions.SERVING_STREAM_NAME,
    redisOutputQueue: String = "result:",
    parallelism: Int = 1000,
    timeWindow: Int = 0,
    countWindow: Int = 56,
    tokenBucketEnabled: Boolean = false,
    tokensPerSecond: Int = 100,
    tokenAcquireTimeout: Int = 100,
    httpsEnabled: Boolean = false,
    httpsKeyStorePath: String = null,
    httpsKeyStorePassword: String = "1234qwer",
    redisSecureEnabled: Boolean = false,
    redissTrustStorePath: String = null,
    redissTrustStorePassword: String = "1234qwer"
)

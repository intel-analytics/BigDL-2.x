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

package com.intel.analytics.zoo.serving.http2

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
import com.fasterxml.jackson.databind.ObjectMapper
import com.intel.analytics.zoo.serving.http.FrontEndApp.{defineServerContext, logger, overallRequestTimer, system, timing}
import com.intel.analytics.zoo.serving.http.PredictionInputFlushMessage
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
      val route = timing("initialize http route")() {
        path("") {
          timing("welcome")(overallRequestTimer) {
            complete("welcome to " + name)
          }
        }
      }
      if (arguments.httpsEnabled) {
          val serverContext = defineServerContext(arguments.httpsKeyStoreToken,
            arguments.httpsKeyStorePath)
          Http().bindAndHandle(route, arguments.interface, port = arguments.securePort,
            connectionContext = serverContext)
          logger.info(s"https started at https://${arguments.interface}:${arguments.securePort}")
      }
      Http().bindAndHandle(route, arguments.interface, arguments.port)
        logger.info(s"http started at http://${arguments.interface}:${arguments.port}")
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
      opt[Int]('s', "securePort")
        .action((x, c) => c.copy(securePort = x))
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
      opt[String]('w', "httpsKeyStoreToken")
        .action((x, c) => c.copy(httpsKeyStoreToken = x))
        .text("https keyStore token")
      opt[Boolean]('s', "redisSecureEnabled")
        .action((x, c) => c.copy(redisSecureEnabled = x))
        .text("redis secure enabled or not")
      opt[Boolean]('s', "httpsEnabled")
        .action((x, c) => c.copy(httpsEnabled = x))
        .text("https enabled or not")
      opt[String]('p', "redissTrustStorePath")
        .action((x, c) => c.copy(redissTrustStorePath = x))
        .text("rediss trustStore path")
      opt[String]('w', "redissTrustStoreToken")
        .action((x, c) => c.copy(redissTrustStoreToken = x))
        .text("rediss trustStore password")
    }

    def defineServerContext(httpsKeyStoreToken: String,
                            httpsKeyStorePath: String): ConnectionContext = {
      val token = httpsKeyStoreToken.toCharArray

      val keyStore = KeyStore.getInstance("PKCS12")
      val keystoreInputStream = new File(httpsKeyStorePath).toURI().toURL().openStream()
      require(keystoreInputStream != null, "Keystore required!")
      keyStore.load(keystoreInputStream, token)

      val keyManagerFactory = KeyManagerFactory.getInstance("SunX509")
      keyManagerFactory.init(keyStore, token)

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
  redisInputQueue: String = Conventions.SERVING_STREAM_DEFAULT_NAME,
  redisOutputQueue: String =
    Conventions.RESULT_PREFIX + Conventions.SERVING_STREAM_DEFAULT_NAME + ":",
  parallelism: Int = 1000,
  timeWindow: Int = 0,
  countWindow: Int = 56,
  tokenBucketEnabled: Boolean = false,
  tokensPerSecond: Int = 100,
  tokenAcquireTimeout: Int = 100,
  httpsEnabled: Boolean = false,
  httpsKeyStorePath: String = null,
  httpsKeyStoreToken: String = "1234qwer",
  redisSecureEnabled: Boolean = false,
  redissTrustStorePath: String = null,
  redissTrustStoreToken: String = "1234qwer"
)


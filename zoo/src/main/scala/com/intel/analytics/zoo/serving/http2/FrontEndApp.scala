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
import akka.http.scaladsl.server.Directives.{complete, extract, path, _}
import akka.pattern.ask
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.codahale.metrics.MetricRegistry
import com.google.common.util.concurrent.RateLimiter
import com.intel.analytics.zoo.pipeline.inference.EncryptSupportive
import com.intel.analytics.zoo.serving.utils.Conventions
import com.fasterxml.jackson.databind.ObjectMapper
import com.intel.analytics.zoo.serving.http.FrontEndApp.{defineServerContext, logger, overallRequestTimer, predictRequestTimer, silent, system, timing}
import com.intel.analytics.zoo.serving.http.{Instances, InstancesPredictionInput, Predictions}
import org.slf4j.LoggerFactory
import com.intel.analytics.zoo.serving.http.Supportive

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

      val servableManager = new ServableManager
      timing("load servable manager")(){
        servableManager.load(arguments.servableManagerPath)
      }
      logger.info("Servable Manager Load Finished")

      val route = timing("initialize http route")() {
        path("") {
          timing("welcome")(overallRequestTimer) {
            complete("welcome to " + name)
          }
        }~pathPrefix("models"){
          concat(
            (get & path(Segments)){
              (segs) => {
                timing("welcome")(overallRequestTimer) {
                  complete("get Model Info Port")
                }
              }}~(post & path(Segments) & extract(_.request.entity.contentType) & entity(as[String])) {
              (segs, contentType, content) => {
              timing("welcome")(overallRequestTimer) {
                if (segs.length != 4 || segs(1) != "versions" || segs(3) != "predict"){
                  throw ServingRuntimeException ("parameter not macth", null)
                }
                val modelName = segs(0); val modelVersion = segs(2)


                complete("get Model Info Port")
              }}
            }
          )
        }
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
    opt[String]('s', "servableManagerPath")
      .action((x, c) => c.copy(servableManagerPath = x))
      .text("servableManager Config Path")

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
}

case class FrontEndAppArguments(
                                 interface: String = "0.0.0.0",
                                 port: Int = 10020,
                                 servableManagerPath: String = "/home/yansu/projects/test.yaml"
                               )

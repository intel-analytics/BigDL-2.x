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
package com.intel.analytics.zoo.serving.grpc


import org.slf4j.LoggerFactory



import akka.actor.typed.ActorSystem
import akka.actor.typed.scaladsl.Behaviors

import akka.http.scaladsl.Http
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse

import scala.concurrent.ExecutionContext
import scala.concurrent.Future
import scala.util.Failure
import scala.util.Success
import scala.concurrent.duration._
import com.typesafe.config.ConfigFactory



object FrontEndGRPC extends Supportive {
  override val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    // important to enable HTTP/2 in ActorSystem's config
      val conf = ConfigFactory.parseString("akka.http.server.preview.enable-http2 = on")
        .withFallback(ConfigFactory.defaultApplication())
      val system = ActorSystem[Nothing](Behaviors.empty, "GreeterServer", conf)
      new GreeterServer(system).run()
  }

  class GreeterServer(system: ActorSystem[_]) {

    def run(): Future[Http.ServerBinding] = {
      implicit val sys = system
      implicit val ec: ExecutionContext = system.executionContext

      val service: HttpRequest => Future[HttpResponse] =
        GreeterServiceHandler(new GreeterServiceImpl(system))

      val bound: Future[Http.ServerBinding] = Http(system)
        .newServerAt(interface = "127.0.0.1", port = 8080)
        .enableHttps(serverHttpContext)
        .bind(service)
        .map(_.addToCoordinatedShutdown(hardTerminationDeadline = 10.seconds))

      bound.onComplete {
        case Success(binding) =>
          val address = binding.localAddress
          println("gRPC server bound to {}:{}", address.getHostString, address.getPort)
        case Failure(ex) =>
          println("Failed to bind gRPC endpoint, terminating system", ex)
          system.terminate()
      }

      bound
    }
  }
}
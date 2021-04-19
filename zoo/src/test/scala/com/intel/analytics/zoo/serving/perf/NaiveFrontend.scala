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
import akka.actor.{Actor, ActorRef, ActorSystem, Cancellable, Props}
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
import org.apache.log4j.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.concurrent.Await
import scala.concurrent.duration.DurationInt

class GetA() extends Actor {
  var requestMap = mutable.Map[String, ActorRef]()
  val logger = LoggerFactory.getLogger(getClass)
  override def receive: Receive = {
    case message: PutEndMessage =>
      logger.info(s"PutEndMessage received from ${sender().path.name} at ${System.currentTimeMillis()}")
      requestMap += (message.actor.path.name -> message.actor)
    case message: DequeueMessage =>
      logger.info(s"DequeueMessage received from ${sender().path.name} at ${System.currentTimeMillis()}, request map is ${requestMap}")
      requestMap.foreach(e => e._2 ! TestOutputMessage("1"))
  }
}
class PutA(getA: ActorRef) extends Actor {
  val logger = LoggerFactory.getLogger(getClass)
  implicit val timeout: Timeout = Timeout(10, TimeUnit.SECONDS)
  implicit val executionContext = context.system.dispatcher
  var master: ActorRef = _
  var output: TestOutputMessage = _
  var c: Cancellable = _
  override def receive: Receive = {
    case message: TestInputMessage =>
      println(s"TestInputMessage received from ${sender().path.name}")
      getA ! PutEndMessage(this.self)
      master = sender()
      logger.info(s"start schedule at ${System.currentTimeMillis()}")
      c = context.system.scheduler.scheduleOnce(10000 millisecond,self, message)
      logger.info(s"cancel schedule at ${System.currentTimeMillis()}")
      sender() ! output
    case message: TestOutputMessage =>
      logger.info(s"TestOutputMessage received from ${sender().path.name}")
      output = message
        c.cancel()
//      val a = Await.result(getA ? PutEndMessage(this.self), timeout.duration).asInstanceOf[String]


  }
}
object NaiveFrontend extends SSupportive with EncryptSupportive {
 val logger = LoggerFactory.getLogger(getClass)

  val name = "analytics zoo web serving frontend"

  implicit val system = ActorSystem("zoo-serving-frontend-system")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit = {
    timing(s"$name started successfully.") {
      val redisGetterName = s"redis-getter"
      val getter = timing(s"$redisGetterName initialized."){
        val redisGetterProps = Props(new GetA())
        system.actorOf(redisGetterProps, name = redisGetterName)
      }
      val redisPutterName = s"redis-putter"
      val putter = timing(s"$redisPutterName initialized.") {
        val redisPutterProps = Props(new PutA(getter))
        system.actorOf(redisPutterProps, name = redisPutterName)
      }

      def put(input: String): Unit = {
        putter ! TestInputMessage(input)
      }
      val route = timing("initialize http route") {
        path("") {
          timing("welcome") {
            complete("welcome to " + name)
          }
        } ~ (post & path("predict") & extract(_.request.entity.contentType) & entity(as[String])) {
          (contentType, content) => {

              try {
                val a = Await.result(putter ? TestInputMessage(content), timeout.duration).asInstanceOf[TestOutputMessage]
                logger.info(a.toString)

//                val result = timing("predict") {
//                  Await.result(putter ? DataInputMessage, timeout.duration).asInstanceOf[String]
//                }
//                logger.info(result.toString)
                complete(200, a.toString())
              }
                catch {
                case e =>
                  val message = e.getMessage
                  val exampleJson = "e"
                  val error = ServingError(s"Wrong content format.\n" +
                    s"Details: ${message}\n" +
                    s"Please refer to examples:\n" +
                    s"$exampleJson\n")
                  complete(400, error.error)
              }
            }
          }

      }

      Http().bindAndHandle(route, "0.0.0.0", 10020)
      system.scheduler.schedule(1000 milliseconds, 1000 milliseconds,
        getter, DequeueMessage())(system.dispatcher)
    }
  }


}
trait SSupportive {
  def timing[T](name: String)(f: => T): T = {
    val begin = System.nanoTime()
    val result = f
    val end = System.nanoTime()
    val cost = (end - begin)
    Logger.getLogger(getClass).info(s"$name time elapsed [${cost / 1e6} ms].")
    result
  }
}
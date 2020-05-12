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

import java.util.concurrent.{CountDownLatch, Executors, TimeUnit}

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.unmarshalling.Unmarshal
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.codahale.metrics.MetricRegistry
import com.intel.analytics.zoo.serving.http.FrontEndApp.metrics
import com.intel.analytics.zoo.serving.http.{JsonUtil, ServingTimerMetrics, Supportive}
import org.slf4j.LoggerFactory

import scala.concurrent.Await
import scala.concurrent.duration.Duration
import scala.io.Source

object MockServingHttpClient extends App with Supportive {

  override val logger = LoggerFactory.getLogger(getClass)

  implicit val system = ActorSystem("zoo-serving-frontend-system")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  val rawStr = "aW1hZ2UgYnl0ZXM="
  val jsonStr =
    """
      |{
      |  "instances": [
      |    {
      |      "image": { "b64": "aW1hZ2UgYnl0ZXM=" },
      |      "caption": "seaside"
      |    },
      |    {
      |      "image": { "b64": "YXdlc29tZSBpbWFnZSBieXRlcw==" },
      |      "caption": "mountains"
      |    }
      |  ]
      |}
      |
    """.stripMargin

  val rawRequest = HttpRequest(
    method = HttpMethods.POST,
    uri = Uri(s"http://localhost:10020/predict"))
    .withEntity(ContentTypes.`text/plain(UTF-8)`, rawStr)

  val wrongRequest = HttpRequest(
    method = HttpMethods.POST,
    uri = Uri(s"http://localhost:10020/predict"))
    .withEntity(ContentTypes.`text/html(UTF-8)`, rawStr)

  val jsonRequest = HttpRequest(
    method = HttpMethods.POST,
    uri = Uri(s"http://localhost:10020/predict"))
    .withEntity(ContentTypes.`application/json`, jsonStr)

  silent(s"single http request")() {
    // val rawResponse = handle(rawRequest)
    // println(s"$rawResponse")

    // val wrongResponse = handle(wrongRequest)
    // println(s"$wrongResponse")

    // val jsonResponse = handle(jsonRequest)
    // println(s"$jsonResponse")

    val metrics = new MetricRegistry
    val timer = metrics.timer("timer")
    val n = 1000
    val nTasks = 1000 * n
    val parallism = 50
    val executorService = Executors.newFixedThreadPool(parallism)
    val countDownLatch = new CountDownLatch(3000)
    var num = 0
    val start = System.currentTimeMillis()
    val results = average(s"$nTasks tasks with $parallism parallism")(nTasks)() {
      val results = List.range(0, nTasks).map(i => {
        val task = new Runnable {
          var result = 0
          override def run(): Unit = {
            val rawResponse = silent("")(timer) {
              handle(rawRequest)
            }
            rawResponse._1 / 100 == 2 match {
              case true => result = 1; countDownLatch.countDown()
              case false => result = 0
            }
            println(s"################ $i: $rawResponse, $result")
          }
        }
        executorService.submit(task)
        task
      })
      executorService.shutdown()
      countDownLatch.await()
      results.map(_.result)
    }
    val end = System.currentTimeMillis()
    val keys = metrics.getTimers().keySet()
    val servingMetrics = keys.toArray.map(key => {
      val timer = metrics.getTimers().get(key)
      ServingTimerMetrics(key.toString, timer)
    }).toList
    println(JsonUtil.toJson(servingMetrics))
    val successed = results.filter(_ == 1).size
    val failed = results.filter(_ == 0).size
    println(s"successed throughput: ${successed * 1000.0 / (end - start)}, $successed, $failed")


    // system.terminate()
  }

  def handle(request: HttpRequest): (Int, String) = {
    val future = timing("send")() {
      Http().singleRequest(request)
    }
    val response = timing("receive")() {
      Await.result(future, Duration.Inf)
    }
    val status = response.status.intValue()
    val entity = entityAsString(response.entity)
    (status, entity)
  }

  def entityAsString(entity: ResponseEntity): String = {
    Await.result(Unmarshal(entity).to[String], Duration.Inf).trim
  }

}

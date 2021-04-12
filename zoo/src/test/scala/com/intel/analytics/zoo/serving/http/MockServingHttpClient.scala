package com.intel.analytics.zoo.serving.http

import java.util.concurrent.TimeUnit

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.unmarshalling.Unmarshal
import akka.stream.ActorMaterializer
import akka.util.Timeout
import org.slf4j.LoggerFactory

import scala.concurrent.Await
import scala.concurrent.duration.Duration

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
      |      "image": "aW1hZ2UgYnl0ZXM=",
      |      "caption": "seaside"
      |    },
      |    {
      |      "image": "YXdlc29tZSBpbWFnZSBieXRlcw==",
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

    val wrongResponse = handle(wrongRequest)
    println(s"$wrongResponse")

    val jsonResponse = handle(jsonRequest)
    println(s"$jsonResponse")
    system.terminate()
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

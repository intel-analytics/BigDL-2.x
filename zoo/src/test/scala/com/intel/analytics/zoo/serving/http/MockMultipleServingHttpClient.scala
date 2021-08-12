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


import java.util.concurrent.TimeUnit

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.unmarshalling.Unmarshal
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.codahale.metrics.{MetricRegistry, Timer}
import org.scalatest.{FlatSpec, Matchers}
import org.slf4j.LoggerFactory

import scala.concurrent.Await
import scala.concurrent.duration.Duration

class MockMultipleServingHttpClient extends FlatSpec with Matchers with App with Supportive {

  override val logger = LoggerFactory.getLogger(getClass)
  // load various model
  implicit val system = ActorSystem("zoo-serving-frontend-system")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)
  val metrics = new MetricRegistry
  val timer: Timer = metrics.timer("test")

  testCaffe()

  def testCaffe() : Unit = {
    val resource = getClass().getClassLoader().getResource("models")
    val modelPath = resource.getPath + "/caffe/test_persist.prototxt"
    val weightPath = resource.getPath + "/caffe/test_persist.caffemodel"
    val features = Array("floatTensor")
    val inferenceModelMetaData = InferenceModelMetaData("caffe", "1.0", modelPath, "Caffe",
      weightPath, 1, "instance", features)
    val inferenceServable = new InferenceModelServable(inferenceModelMetaData, timer)

    inferenceServable.load()
    val content =
      """{
  "instances" : [ {
    "floatTensor" : [ [ [0.6804766, 0.30136853, 0.17394465, 0.44770062, 0.20275897] ] ]
  } ]
}"""
    require(inferenceServable.predict(content).mkString == "1")
  }

}

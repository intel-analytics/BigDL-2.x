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

import java.nio.file.{Files, Paths}
import java.util.{Base64, UUID}

import com.intel.analytics.zoo.serving.http._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.immutable.TreeMap
import scala.collection.mutable

class FrontendDomainsSpec extends FlatSpec with Matchers with BeforeAndAfter with Supportive {

  "ServingError" should "serialized as json" in {
    val message = "contentType not supported"
    val error = ServingError(message)
    error.toString should include (s""""error" : "$message"""")
  }

  "Feature" should "serialized and deserialized as json" in {
    val image1 = new ImageFeature("aW1hZ2UgYnl0ZXM=")
    val image2 = new ImageFeature("YXdlc29tZSBpbWFnZSBieXRlcw==")
    val image3Path = getClass().getClassLoader()
      .getResource("imagenet/n02110063/n02110063_15462.JPEG").getFile()
    val byteArray = Files.readAllBytes(Paths.get(image3Path))
    val image3 = new ImageFeature(Base64.getEncoder().encodeToString(byteArray))
    val instance3 = mutable.LinkedHashMap[String, Any]("image" -> image3, "caption" -> "dog")
    val inputs = Instances(List.range(0, 2).map(i => instance3))
    val json = timing("serialize")() {
      JsonUtil.toJson(inputs)
    }
    println(json)
    val obj = timing("deserialize")() {
      JsonUtil.fromJson(classOf[Instances], json)
    }
    obj.instances.size should be (2)
  }

  "BytesPredictionInput" should "works well" in {
    val bytesStr = "aW1hZ2UgYnl0ZXM="
    val input = BytesPredictionInput(bytesStr)
    input.toHash().get("image") should equal(bytesStr)
  }

  "PredictionOutput" should "works well" in {
    val uuid = UUID.randomUUID().toString
    val result = "mock-result"
    val out = PredictionOutput(uuid, result)
    out.uuid should be (uuid)
    out.result should be (result)
  }

  val instancesJson = """{
                        |"instances": [
                        |   {
                        |     "tag": "foo",
                        |     "signal": [1, 2, 3, 4, 5],
                        |     "sensor": [[1, 2], [3, 4]]
                        |   },
                        |   {
                        |     "tag": "bar",
                        |     "signal": [3, 4, 1, 2, 5],
                        |     "sensor": [[4, 5], [6, 8]]
                        |   }
                        |]
                        |}
                        |""".stripMargin
  "Instances" should "works well" in {
    val instances = JsonUtil.fromJson(classOf[Instances], instancesJson)
    instances.instances.size should be (2)

    val intScalar = 12345
    val floatScalar = 3.14159
    val stringScalar = "hello, world. hello, arrow."
    val intTensor = List(1, 2, 3, 4, 5)
    val floatTensor = List(0.5f, 0.7f, 4.678f, 8.9f, 9.8765f)
    val stringTensor = List("come", "on", "united")
    val intTensor2 = List(List(1, 2), List(3, 4), List(5, 6))
    val floatTensor2 =
      List(
        List(
          List(.2f, .3f),
          List(.5f, .6f)),
        List(
          List(.2f, .3f),
          List(.5f, .6f)))
    val stringTensor2 =
      List(
        List(
          List(
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united")),
          List(
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"))
        ),
        List(
          List(
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united")),
          List(
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"))
        )
      )
    val instance = mutable.LinkedHashMap(
      "intScalar" -> intScalar,
      "floatScalar" -> floatScalar,
      "stringScalar" -> stringScalar,
      "intTensor" -> intTensor,
      "floatTensor" -> floatTensor,
      "stringTensor" -> stringTensor,
      "intTensor2" -> intTensor2,
      "floatTensor2" -> floatTensor2,
      "stringTensor2" -> stringTensor2
    )
    val instances2 = Instances(instance)
    val json2 = JsonUtil.toJson(instances2)
    println(json2)
    val instances3 = JsonUtil.fromJson(classOf[Instances], json2)
    val tensors = instances3.constructTensors()
    val schemas = instances3.makeSchema(tensors)
    println(tensors)
    println(schemas)

    val (shape1, data1) = Instances.transferListToTensor(intTensor)
    shape1.reduce(_*_) should be (data1.size)
    val (shape2, data2) = Instances.transferListToTensor(intTensor2)
    shape2.reduce(_*_) should be (data2.size)
    val (shape3, data3) = Instances.transferListToTensor(floatTensor2)
    shape3.reduce(_*_) should be (data3.size)
    val (shape4, data4) = Instances.transferListToTensor(stringTensor2)
    shape4.reduce(_*_) should be (data4.size)

    val arrowBytes = instances3.toArrow()
    println(arrowBytes)
    println(arrowBytes.length)

    Instances.fromArrow(arrowBytes)
  }
}

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
    val instance3 = Map("image" -> image3, "caption" -> "dog")
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
}

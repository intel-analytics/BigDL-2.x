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

package com.intel.analytics.zoo.serving.models

import org.scalatest.{FlatSpec, Matchers}
import sys.process._

class OpenVINOModelSpec extends FlatSpec with Matchers {
  "OpenVINO ResNet50" should "work" in {
    ("wget -O /tmp/serving_val.tar http://10.239.45.10:8081" +
      "/repository/raw/analytics-zoo-data/imagenet_1k.tar").!
    "tar -xvf /tmp/serving_val.tar -C /tmp/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-base64"
    val modelPath = "/tmp/"


  }

}

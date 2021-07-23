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

package com.intel.analytics.zoo.serving.preprocessing

import com.intel.analytics.zoo.serving.utils.ConfigParser
import com.intel.analytics.zoo.serving.{ClusterServing, TestUtils}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ImageProcessingSpec extends FlatSpec with Matchers with BeforeAndAfter{
  before {
    ClusterServing.helper = new ConfigParser(
      getClass().getClassLoader().getResource("serving").getPath
        + "/image-test-config.yaml").loadConfig()
  }

  "image resize" should "work" in {
    val imageB64 = TestUtils.getStrFromResourceFile("image-3_224_224-jpg-base64")
    val preProcessing = new PreProcessing()
    val tensor = preProcessing.decodeImage(imageB64)
    assert(tensor.size()(0) == 3)
    assert(tensor.size()(1) == 225)
    assert(tensor.size()(2) == 225)
  }

}

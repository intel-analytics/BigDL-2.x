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

package com.intel.analytics.zoo.pipeline.api

import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers.Dense

class NetSpec extends ZooSpecHelper{

  "invokeMethod with params" should "work properly" in {
    assert(Dense[Float](3).invokeMethod("build", Shape(2, 5)) == Shape(2, 3))
  }

  "invokeMethod without params" should "work properly" in {
    assert(Dense[Float](3).invokeMethod("isBuilt") == false)
  }

  "invokeMethod for Seq" should "work properly" in {
    Dense[Float](3).invokeMethodForSeq("excludeInvalidLayers", Seq(Dense[Float](3)))
  }

}

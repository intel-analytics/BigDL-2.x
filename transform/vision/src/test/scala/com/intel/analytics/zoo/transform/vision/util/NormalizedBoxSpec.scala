/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.transform.vision.util

import org.scalatest.{FlatSpec, Matchers}

class NormalizedBoxSpec extends FlatSpec with Matchers{
  "scaleBBox" should "work properly" in {
    val bbox = NormalizedBox(1, 4, 5, 6)
    val scaled = new NormalizedBox()
    bbox.scaleBox(1.0f / 4, 1.0f / 2, scaled)

    scaled should be (NormalizedBox(0.5f, 1, 2.5f, 1.5f))
  }
}

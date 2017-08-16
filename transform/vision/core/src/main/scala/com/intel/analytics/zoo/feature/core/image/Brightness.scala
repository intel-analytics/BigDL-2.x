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

package com.intel.analytics.zoo.feature.core.image

import com.intel.analytics.zoo.feature.core.util.MatWrapper

/**
 * adjust the image brightness
 * @param delta brightness parameter
 * if delta > 0, increase the brightness
 * if delta < 0, decrease the brightness
 */
class Brightness(delta: Float)
  extends FeatureTransformer {
  override def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean = {
    Brightness.transform(input, output, delta)
    true
  }
}

object Brightness {
  def apply(delta: Float): Brightness
  = new Brightness(delta)

  def transform(input: MatWrapper, output: MatWrapper, delta: Float): MatWrapper = {
    if (delta != 0) {
      input.convertTo(output, -1, 1, delta)
    } else {
      if (input != output) input.copyTo(output)
    }
    output
  }
}

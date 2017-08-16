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

import java.util

import com.intel.analytics.zoo.feature.core.util.MatWrapper
import org.opencv.core.{Core, Mat}

/**
 * random change the channel of an image
 * @param randomOrderProb the probability to reorder the channels
 */
class ChannelOrder(randomOrderProb: Float)
  extends FeatureTransformer {
  override def transform(input: MatWrapper, output: MatWrapper, feature: Feature): Boolean = {
    randomOperation(ChannelOrder.transform, input, output, randomOrderProb)
  }

}

object ChannelOrder {
  def apply(randomOrderProb: Float): ChannelOrder
  = new ChannelOrder(randomOrderProb)

  def transform(input: MatWrapper, output: MatWrapper): MatWrapper = {
    // Split the image to 3 channels.
    val channels = new util.ArrayList[Mat]()
    Core.split(output, channels)
    // Shuffle the channels.
    util.Collections.shuffle(channels)
    Core.merge(channels, output)
    // release memory
    (0 until channels.size()).foreach(channels.get(_).release())
    output
  }
}

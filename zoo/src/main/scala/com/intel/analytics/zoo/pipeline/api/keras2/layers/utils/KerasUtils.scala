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

package com.intel.analytics.zoo.pipeline.api.keras2.layers.utils


import com.intel.analytics.bigdl.nn.abstractnn.DataFormat


object KerasUtils {

  def toBigDLFormat(dataFormat: String): DataFormat = {
    require(dataFormat.toLowerCase() == "channels_" +
      "first" || dataFormat.toLowerCase() == "channels_last",
      s"Data Format must be either channels_first or " +
      s"channels_last, but got ${dataFormat.toLowerCase()}")
    dataFormat.toLowerCase() match {
      case "channels_last" => DataFormat.NHWC
      case "channels_first" => DataFormat.NCHW
    }
  }

  def toBigDLFormat5D(dataFormat: String): String = {
    require(dataFormat.toLowerCase() == "channels_fir" +
      "st" || dataFormat.toLowerCase() == "channels_last",
      s"Data Format must be either channels_first or" +
        s" channels_last, but got ${dataFormat.toLowerCase()}")
    dataFormat.toLowerCase() match {
      case "channels_last" => "CHANNEL_LAST"
      case "channels_first" => "CHANNEL_FIRST"
    }
  }

}


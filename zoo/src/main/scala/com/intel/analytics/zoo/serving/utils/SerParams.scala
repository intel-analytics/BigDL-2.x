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

package com.intel.analytics.zoo.serving.utils

import java.text.SimpleDateFormat

class SerParams(helper: ClusterServingHelper, loadModel: Boolean = true) extends Serializable {
  var redisHost = helper.redisHost
  var redisPort = helper.redisPort.toInt
  val coreNum = helper.coreNum
  val filter = helper.filter
  val chwFlag = helper.chwFlag
  val inferenceMode = helper.inferenceMode
  val dataShape = helper.dataShape
  val modelType = helper.modelType
  val modelDir = helper.modelDir
  val lastModified = FileUtils.getLastModified(helper.modelDir)
  val sdf = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss")
  println(s"loading params, time is ${sdf.format(lastModified)}")
  var model = if (loadModel) {
    helper.loadInferenceModel()
  } else {
    null
  }
}

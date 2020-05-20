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

class SerParams(helper: ClusterServingHelper) extends Serializable {
  var redisHost = helper.redisHost
  var redisPort = helper.redisPort.toInt
  val coreNum = helper.coreNum
  val filter = helper.filter
  val chwFlag = helper.chwFlag
  val C = helper.dataShape(0)(0)
  val H = helper.dataShape(0)(1)
  val W = helper.dataShape(0)(2)
  val modelType = helper.modelType
  val model = helper.loadInferenceModel()
}

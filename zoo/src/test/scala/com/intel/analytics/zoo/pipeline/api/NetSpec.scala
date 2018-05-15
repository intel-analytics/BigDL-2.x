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

import com.intel.analytics.bigdl.{Module, NetUtils}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers.Dense
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

class NetSpec extends ZooSpecHelper{

  "invokeMethod set inputShape" should "work properly" in {
    KerasUtils.invokeMethod(Dense[Float](3), "_inputShapeValue_$eq", Shape(2, 3))
  }

  "Load Caffe" should "work" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "caffe"
    val model = Net.loadCaffe[Float](
      s"$path/test_persist.prototxt", s"$path/test_persist.caffemodel")
    val newModel = model.newGraph("ip")
    NetUtils.getOutputs(newModel).head.element.getName() should be ("ip")
  }

  "Load bigdl" should "work" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "bigdl"
    val model = Net.loadBigDL[Float](s"$path/bigdl_lenet.model")
    val newModel = model.newGraph("reshape2")
    NetUtils.getOutputs(newModel).head.element.getName() should be ("reshape2")
  }

  "Load tensorflow" should "work" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "tensorflow"
    val model = Net.loadTF[Float](s"$path/lenet.pb", Seq("Placeholder"), Seq("LeNet/fc4/BiasAdd"))
    val newModel = model.newGraph("LeNet/fc3/Relu")
    NetUtils.getOutputs(newModel).head.element.getName() should be ("LeNet/fc3/Relu")
  }

}

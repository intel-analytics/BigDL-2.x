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

import com.intel.analytics.zoo.serving.PreProcessing
import com.intel.analytics.zoo.serving.arrow.ArrowDeserializer
import com.intel.analytics.zoo.serving.engine.{ClusterServingInference, ModelHolder}
import com.intel.analytics.zoo.serving.utils.{ClusterServingHelper, SerParams}
import org.scalatest.{FlatSpec, Matchers}

import sys.process._

class OpenVINOModelSpec extends FlatSpec with Matchers {
  "OpenVINO ResNet50" should "work" in {
    ("wget -O /tmp/serving_val.tar http://10.239.45.10:8081" +
      "/repository/raw/analytics-zoo-data/imagenet_1k.tar").!
    "tar -xvf /tmp/serving_val.tar -C /tmp/".!
    val resource = getClass().getClassLoader().getResource("serving")
    val dataPath = resource.getPath + "/image-3_224_224-base64"
    val b64string = scala.io.Source.fromFile(dataPath).mkString

    val helper = new ClusterServingHelper()
    helper.modelType = "openvino"
    helper.weightPath = "/tmp/serving_val_openvino_res50/resnet_v1_50.bin"
    helper.defPath = "/tmp/serving_val_openvino_res50/resnet_v1_50.xml"
    ModelHolder.model = helper.loadInferenceModel()
    ("rm -rf /tmp/imagenet_1k*").!
    "rm -rf /tmp/serving_val_*".!
    "rm -rf /tmp/config.yaml".!
    val params = new SerParams(helper)
    val inference = new ClusterServingInference(new PreProcessing(params.chwFlag),
      params.modelType, "", params.coreNum, params.resize)
    val in = List(("1", b64string), ("2", b64string), ("3", b64string))
    val postProcessed = inference.multiThreadPipeline(in)

    postProcessed.foreach(x => {
      val result = ArrowDeserializer.getArray(x._2)
      require(result(0)._1.length == 1000, "result length wrong")
      require(result(0)._2.length == 1, "result shape wrong")
    })
  }

}

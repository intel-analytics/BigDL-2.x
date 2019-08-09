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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{CAddTable, SpatialCrossMapLRN}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.bigdl.utils.caffe.{CaffeLoader => BigDLCaffeLoader}
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Input, KerasLayerWrapper}
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Model => ZModel}

import scala.util.Random

class NetSpec extends ZooSpecHelper{

  "invokeMethod set inputShape" should "work properly" in {
    KerasUtils.invokeMethod(Dense[Float](3), "_inputShapeValue_$eq", Shape(2, 3))
  }

  "Zoo CaffeLoader and BigDL CaffeLoader" should "have the same result" in {

    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "caffe"
    val ww = s"$path/test_persist.caffemodel"
    val dd = s"$path/test_persist.prototxt"
    // val ww = "/home/litchy/bvlc.caffemodel"
    // val dd = "/home/litchy/deploy_overlap.prototxt"

    val bigDlModel = BigDLCaffeLoader.loadCaffe[Float](dd, ww)._1

    val zooModel = Net.loadCaffe[Float](dd, ww)

    val inputTensor = Tensor[Float](1, 3, 224, 224).apply1(e => Random.nextFloat())
    val zooResult = zooModel.forward(inputTensor)
    val bigDlResult = bigDlModel.forward(inputTensor)
    zooResult should be (bigDlResult)
  }

  "Load Caffe model" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "caffe"
    val model = Net.loadCaffe[Float](
      s"$path/test_persist.prototxt", s"$path/test_persist.caffemodel")
    val newModel = model.newGraph("ip")
    newModel.outputNodes.head.element.getName() should be("ip")
  }

  "createTmpFile" should "work properly" in {
    val tmpFile = ZooSpecHelper.createTmpFile()
    print(tmpFile)
  }

  "Load Keras-style Analytics Zoo model" should "work properly" in {
    val input = Input[Float](inputShape = Shape(3, 5))
    val d = Dense[Float](7).setName("dense1").inputs(input)
    val model = ZModel[Float](input, d)

    val tmpFile = createTmpFile()
    val absPath = tmpFile.getAbsolutePath
    model.saveModule(absPath, overWrite = true)

    val reloadedModel = Net.load[Float](absPath)
      .asInstanceOf[KerasNet[Float]]

    val inputTensor = Tensor[Float](2, 3, 5).rand()
    compareOutputAndGradInput(
      model.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      reloadedModel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], inputTensor)
  }

  "Load BigDL model" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "bigdl"
    val model = Net.loadBigDL[Float](s"$path/bigdl_lenet.model")
    val newModel = model.newGraph("reshape2")
    newModel.outputNodes.head.element.getName() should be ("reshape2")
  }

  "Load Torch model" should "work properly" in {
    val layer = new SpatialCrossMapLRN[Float](5, 1.0, 0.75, 1.0)

    val tmpFile = java.io.File.createTempFile("module", ".t7")
    val absolutePath = tmpFile.getAbsolutePath
    layer.saveTorch(absolutePath, true)

    val reloadedModel = Net.loadTorch[Float](absolutePath)

    val inputTensor = Tensor[Float](16, 3, 224, 224).rand()
    compareOutputAndGradInput(
      layer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      reloadedModel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], inputTensor)
  }

  "Load Tensorflow model" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "tensorflow"
    val model = Net.loadTF[Float](s"$path/frozen_inference_graph.pb",
      Seq("Placeholder"), Seq("dense_1/Sigmoid"))
    val newModel = model.newGraph("dense/Relu")
    newModel.outputNodes.head.element.getName() should be ("dense/Relu")
  }

  "net load model" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "zoo_keras"

    val seq = Net.load[Float](s"$path/small_seq.model")
    seq.forward(Tensor[Float](2, 3, 5).rand())

    val model = Net.load[Float](s"$path/small_model.model")
    model.forward(Tensor[Float](2, 3, 5).rand())
  }

  "connect variable " should "work properly" in {
    def createOldModel(): AbstractModule[Activity, Activity, Float] = {
      val input1 = Input[Float](inputShape = Shape(3))
      val input2 = Input[Float](inputShape = Shape(3))
      val sum = new KerasLayerWrapper[Float](
        CAddTable[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]])
        .inputs(Array(input1, Dense[Float](3).inputs(input2)))
      ZModel[Float](input = Array(input1, input2), output = sum)
    }
    val input1 = Variable[Float](inputShape = Shape(3))
    val input2 = Variable[Float](inputShape = Shape(3))
    val diff = input1 + Dense[Float](3).from(input2)
    val model = ZModel[Float](input = Array(input1, input2), output = diff)
    val inputValue = Tensor[Float](1, 3).randn()
    val oldModel = createOldModel()
    val out = model.forward(T(inputValue, inputValue)).toTensor[Float]
    val oldOut = oldModel.forward(T(inputValue, inputValue)).toTensor[Float]
    out.almostEqual(oldOut, 1e-4)
  }

  "Load Tensorflow model from path" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("tfnet")
    val model = Net.loadTF[Float](resource.getPath)
    val result = model.forward(Tensor[Float](2, 4).rand())
    result.toTensor[Float].size() should be (Array(2, 2))
  }
}

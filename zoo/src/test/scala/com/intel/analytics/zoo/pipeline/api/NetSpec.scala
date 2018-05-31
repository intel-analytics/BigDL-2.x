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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.SpatialCrossMapLRN
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Sequential, Model => ZModel}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Input}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

class NetSpec extends ZooSpecHelper{

  "invokeMethod set inputShape" should "work properly" in {
    KerasUtils.invokeMethod(Dense[Float](3), "_inputShapeValue_$eq", Shape(2, 3))
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
    val model = Net.loadTF[Float](s"$path/lenet.pb", Seq("Placeholder"), Seq("LeNet/fc4/BiasAdd"))
    val newModel = model.newGraph("LeNet/fc3/Relu")
    newModel.outputNodes.head.element.getName() should be ("LeNet/fc3/Relu")
  }

  "net load model" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "zoo_keras"

    val seq = Net.load[Float](s"$path/small_seq.model")
    seq.forward(Tensor[Float](2, 3, 5).rand())

    val model = Net.load[Float](s"$path/small_model.model")
    model.forward(Tensor[Float](2, 3, 5).rand())
  }
}

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

package com.intel.analytics.zoo.pipeline.inference

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.{PythonInterpreter, PythonInterpreterTest}
import com.intel.analytics.zoo.core.TFNetNative
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.net.TorchModel
import org.apache.log4j.{Level, Logger}

import scala.language.postfixOps


@PythonInterpreterTest
class PyTorchModelSpec extends ZooSpecHelper with InferenceSupportive {

  var model: InferenceModel = _
  var model2: InferenceModel = _
  val currentNum = 10
  var modelPath: String = _
  var modelPathMulti: String = _

  protected def ifskipTest(): Unit = {
    // Skip unitest if environment is not ready, PYTHONHOME should be set in environment
    if (System.getenv("PYTHONHOME") == null) {
      cancel("Please export PYTHONHOME before this test.")
    } else {
      logger.info(s"use python home: ${System.getenv("PYTHONHOME")}")
      Logger.getLogger(PythonInterpreter.getClass()).setLevel(Level.DEBUG)
      // Load TFNet before create interpreter, or the TFNet will throw an OMP error #13
      TFNetNative.isLoaded
    }
  }

  val resnetModel =
    s"""
       |import torch
       |import torchvision.models as models
       |from zoo.pipeline.api.torch import zoo_pickle_module
       |
       |model = models.resnet18(pretrained = True)
       |print("after load model and before save the model's path")
       |torch.save(model, "$modelPath", pickle_module=zoo_pickle_module)
       |print("before model two")
       |modelMulti = models.resnet50(pretrained = True)
       |print("before save model two")
       |torch.save(modelMulti, "$modelPathMulti", pickle_module=zoo_pickle_module)
       |print("after model code")
       |print("before load model")
       |""".stripMargin

  protected def beforeAll()  {
    println("before get path")
    model = new InferenceModel(currentNum) { }
    model2 = new InferenceModel(currentNum) { }
    modelPath = ZooSpecHelper.createTmpFile().getAbsolutePath()
    modelPathMulti = ZooSpecHelper.createTmpFile().getAbsolutePath()
    PythonInterpreter.exec(resnetModel)
  }

  protected def afterAll() {
    model.doRelease()
    model2.doRelease()
  }

  "PyTorch Model" should "be loaded" in {
    beforeAll()
    ifskipTest()

    val modelone = TorchModel.loadModel(modelPath)
    modelone.evaluate()

    val PyTorchModel = ModelLoader.loadFloatModelForPyTorch(modelPath)
    PyTorchModel.evaluate()
    val metaModel = makeMetaModel(PyTorchModel)
    val floatFromPyTorch = new FloatModel(PyTorchModel, metaModel, true)
    println(floatFromPyTorch)
    floatFromPyTorch shouldNot be(null)

    model.doLoadPyTorch(modelPath)
    println(model)
    model shouldNot be(null)

    val modelBytes = Files.readAllBytes(Paths.get(modelPath))
    model2.doLoadPyTorch(modelBytes)
    println(model2)
    model2 shouldNot be(null)

    val threads = List.range(0, currentNum).map(i => {
      new Thread() {
        override def run(): Unit = {
          model.doLoadPyTorch(modelPath)
          println(model)
          model shouldNot be(null)

          model2.doLoadPyTorch(modelBytes)
          println(model2)
          model2 shouldNot be(null)
        }
      }
    })
    threads.foreach(_.start())
    threads.foreach(_.join())


    afterAll()
  }

  "PyTorch Model" should "do predict" in {
    beforeAll()
    ifskipTest()

    val inputTensor = Tensor[Float](1, 3, 224, 224).rand()
    model.doLoadPyTorch(modelPath)
    model2.doLoadPyTorch(modelPathMulti)
    val results = model.doPredict(inputTensor)
    println(results)
    val threads = List.range(0, currentNum).map(i => {
      new Thread() {
        override def run(): Unit = {
          val r = model.doPredict(inputTensor)
          println(r)
          r should be (results)

          val r2 = model2.doPredict(inputTensor)
          println(r2)
          r2 should be (results)
        }
      }
    })
    threads.foreach(_.start())
    threads.foreach(_.join())

    afterAll()
  }

  "PyTorch Models' weights" should "be the same" in {
    beforeAll()
    ifskipTest()

    val threads = List.range(0, currentNum).map(i => {
      new Thread() {
        override def run(): Unit = {
          val PyTorchModel = ModelLoader.loadFloatModelForPyTorch(modelPath)
          PyTorchModel.evaluate()
          var weightsOfTorchModel = PyTorchModel.parameters()
          val metaModel = makeMetaModel(PyTorchModel)
          val floatFromPyTorch = new FloatModel(PyTorchModel, metaModel, true)
          val weightsOfFloatModel = floatFromPyTorch.model.parameters()
          println(floatFromPyTorch)
          floatFromPyTorch shouldNot be(null)

          if(weightsOfTorchModel == weightsOfFloatModel)
            {
              println("weights of torch == weights of float")
            }
            else {
            println("weights of torch != weights of float")
          }

          // multi model test
          val PyTorchModel2 = ModelLoader.loadFloatModelForPyTorch(modelPathMulti)
          PyTorchModel2.evaluate()
          val weightsOfTorchModel2 = PyTorchModel2.parameters()
          val metaModel2 = makeMetaModel(PyTorchModel2)
          val floatFromPyTorch2 = new FloatModel(PyTorchModel2, metaModel2, true)
          val weightsOfFloatModel2 = floatFromPyTorch2.model.parameters()
          println(floatFromPyTorch2)
          floatFromPyTorch2 shouldNot be(null)

          if(weightsOfTorchModel2 == weightsOfFloatModel2)
            {
              println("weights of torch2 == weights of float2")
            }
            else {
            println("weights of torch2 != weights of float2")
          }

          val modelBytes = Files.readAllBytes(Paths.get(modelPath))
          val PyTorchModel3 = ModelLoader.loadFloatModelForPyTorch(modelBytes)
          PyTorchModel3.evaluate()
          val weightsOfTorchModel3 = PyTorchModel3.parameters()
          val metaModel3 = makeMetaModel(PyTorchModel3)
          val floatFromPyTorch3 = new FloatModel(PyTorchModel3, metaModel3, true)
          val weightsOfFloatModel3 = floatFromPyTorch3.model.parameters()
          println(floatFromPyTorch3)
          floatFromPyTorch3 shouldNot be(null)

          if(weightsOfTorchModel3 == weightsOfFloatModel3)
          {
            println("weights of torch3 == weights of float3")
          }
          else {
            println("weights of torch3 != weights of float3")
          }

          // multi model test
          println("before load fourth model")
          val modelBytes2 = Files.readAllBytes(Paths.get(modelPathMulti))
          val PyTorchModel4 = ModelLoader.loadFloatModelForPyTorch(modelBytes2)
          PyTorchModel4.evaluate()
          val weightsOfTorchModel4 = PyTorchModel4.parameters()
          val metaModel4 = makeMetaModel(PyTorchModel4)
          val floatFromPyTorch4 = new FloatModel(PyTorchModel4, metaModel4, true)
          val weightsOfFloatModel4 = floatFromPyTorch4.model.parameters()
          println(floatFromPyTorch4)
          floatFromPyTorch4 shouldNot be(null)

          if(weightsOfTorchModel4 == weightsOfFloatModel4)
          {
            println("weights of torch4 == weights of float4")
          }
          else {
            println("weights of torch4 != weights of float4")
          }

          // weights of path vs weights of bytes
          if(weightsOfTorchModel2 == weightsOfTorchModel4)
          {
            println("weights of path == weights of bytes")
          }
          else {
            println("weights of path != weights of bytes")
          }

        }
      }
    })
    threads.foreach(_.start())
    threads.foreach(_.join())

    afterAll()
  }

}

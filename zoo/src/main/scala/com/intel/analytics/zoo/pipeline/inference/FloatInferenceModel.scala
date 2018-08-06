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

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}

import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.JavaConverters._
import java.util.{List => JList}
import java.lang.{Float => JFloat}
import java.lang.{Integer => JInt}

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T}

class FloatInferenceModel(var model: AbstractModule[Activity, Activity, Float])
  extends InferenceSupportive with Serializable {

  @deprecated
  def predict(input: JList[JFloat], shape: JList[JInt]): JList[JFloat] = {
    timing("model predict") {
      val input_arr = new Array[Float](input.size())
      for (i <- 0 until input.size()) {
        input_arr(i) = input.get(i)
      }
      val shape_arr = new Array[Int](shape.size())
      for (i <- 0 until shape.size()) {
        shape_arr(i) = shape.get(i)
      }
      val result = model.forward(Tensor[Float](input_arr, shape_arr))
      result.asInstanceOf[Tensor[Float]].toArray().toList.asJava.asInstanceOf[JList[JFloat]]
    }
  }

  def predict(inputs: JList[JList[JTensor]]): JList[JList[JTensor]] = {
    timing(s"model predict for batch ${inputs.size()}") {
      val batchSize = inputs.size()
      val outputResults: Array[JList[JTensor]] = new Array[JList[JTensor]](batchSize)
      var i = 0
      while (i < batchSize) {
        val inputList = inputs.get(i)
        val inputLength = inputList.size()
        val result: Activity = inputLength match {
          case 0 =>
            throw new InferenceRuntimeException("input of JList[JTensor] cannot be 0 length")
          case 1 =>
            val input = inputList.get(0)
            val inputData = input.getData
            val inputShape = input.getShape
            val inputTensor = Tensor[Float](inputData, inputShape)
            model.forward(inputTensor)
          case _ =>
            var j = 0
            val inputTable = T()
            while(j < inputLength) {
              val input = inputList.get(j)
              val inputData = input.getData
              val inputShape = input.getShape
              val inputTensor = Tensor[Float](inputData, inputShape)
              inputTable.insert(inputTensor)
              j += 1
            }
            model.forward(inputTable)
        }
        val outputs: Seq[JTensor] = result.isTensor match {
          case true =>
            val outputTensor = result.asInstanceOf[Tensor[Float]]
            Seq(transferTensorToJTensor(outputTensor))
          case false =>
            val outputTable = result.toTable
            outputTable.toSeq[Tensor[Float]].map(t =>
              transferTensorToJTensor(t)
            )
        }
        outputResults(i) = outputs.asJava
        i += 1
      }
      outputResults.toList.asJava
    }
  }

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.writeObject(model)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    System.setProperty("bigdl.localMode", System.getProperty("bigdl.localMode", "true"))
    System.setProperty("bigdl.coreNumber", System.getProperty("bigdl.coreNumber", "1"))
    Engine.init
    model = (in.readObject().asInstanceOf[AbstractModule[Activity, Activity, Float]])
    model.evaluate()
  }

  override def toString: String = s"FloatInferenceModel($model)"
}

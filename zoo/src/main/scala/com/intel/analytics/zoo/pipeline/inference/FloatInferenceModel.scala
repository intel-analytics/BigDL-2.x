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


import com.intel.analytics.bigdl.utils.Engine

class FloatInferenceModel(
  var model: AbstractModule[Activity, Activity, Float],
  @transient var predictor: LocalPredictor[Float]) extends InferenceSupportive with Serializable {

  @deprecated
  def predict(input: JList[JFloat], shape: JList[JInt]): JList[JFloat] = {
    timing("model predict") {
      val sample = transferInputToSample(input, shape)
      val result = predictor.predict(Array(sample))
      require(result.length == 1, "only one input, should get only one prediction")
      result(0).asInstanceOf[Tensor[Float]].toArray().toList.asJava.asInstanceOf[JList[JFloat]]
    }
  }

  def predict(inputs: JList[JTensor]): JList[JList[JTensor]] = {
    timing(s"model predict for batch ${inputs.size()}") {

      var i = 0
      val length = inputs.size()
      val samples = new Array[Sample[Float]](length)
      while (i < length) {
        val input = inputs.get(i)
        val inputData = input.getData
        val inputShape = input.getShape
        val sample = transferInputToSample(inputData, inputShape)
        samples(i) = sample
        i += 1
      }

      val results: Array[Activity] = timing("predictor predict time") {
        predictor.predict(samples)
      }
      val outputResults: Array[JList[JTensor]] = results.map(result => {
        val outputs: List[JTensor] = result.isTensor match {
          case true =>
            val outputTensor = result.asInstanceOf[Tensor[Float]]
            List(transferTensorToJTensor(outputTensor))
          case false =>
            val outputTable = result.toTable
            outputTable.keySet.map(key => {
              val outputTensor = outputTable.get(key).get.asInstanceOf[Tensor[Float]]
              transferTensorToJTensor(outputTensor)
            }).toList
        }
        outputs.asJava.asInstanceOf[JList[JTensor]]
      })
      outputResults.toList.asJava.asInstanceOf[JList[JList[JTensor]]]
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
    predictor = LocalPredictor(model = model, batchPerCore = 1)
  }

  override def toString : String = s"FloatInferenceModel($model, $predictor)"
}

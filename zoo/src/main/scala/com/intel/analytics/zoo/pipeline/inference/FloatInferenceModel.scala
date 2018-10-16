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
import java.lang.{Float => JFloat, Integer => JInt}
import java.util.{List => JList}

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, Table}

import scala.collection.JavaConverters._

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
      require(batchSize > 0, "inputs size should > 0")

      val inputActivity = transferListOfActivityToActivityOfBatch(inputs, batchSize)
      val result: Activity = model.forward(inputActivity)

      val outputs = result.isTensor match {
        case true =>
          val outputTensor = result.toTensor[Float]
          transferBatchTensorToJListOfJListOfJTensor(outputTensor, batchSize)
        case false =>
          val outputTable: Table = result.toTable
          transferBatchTableToJListOfJListOfJTensor(outputTable, batchSize)
      }
      outputs
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

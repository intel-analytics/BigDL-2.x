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

import com.intel.analytics.bigdl.optim.LocalPredictor
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor

import scala.collection.JavaConverters._
import java.util.{List => JList}
import java.lang.{Float => JFloat}
import java.lang.{Integer => JInt}

case class FloatInferenceModel(
  model: AbstractModule[Activity, Activity, Float],
  predictor: LocalPredictor[Float]) extends InferenceSupportive {

  def predict(inputs: JList[JTensor]): JList[JList[JTensor]] = {
    timing(s"model predict for batch ${inputs.size()}") {
      val samples = inputs.asScala.map(input => {
        val inputData = input.getData
        val inputShape = input.getShape
        val sample = transferInputToSample(inputData,
          inputShape.asScala.toArray.map(_.asInstanceOf[Int]))
        sample
      }).toArray
      val results: Array[Activity] = predictor.predict(samples)
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
}

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

case class FloatInferenceModel(
  model: AbstractModule[Activity, Activity, Float],
  predictor: LocalPredictor[Float]) extends InferenceSupportive {

  def predict2d(input: JList[JList[JFloat]]): JList[JList[JFloat]] = {
    timing(s"predict for input") {
      val samples = transfer2dInputToSampleArray(input)
      val result = predictor.predict(samples).toList
        .map(_.asInstanceOf[Tensor[Float]].toArray().toList)
        .asJava.asInstanceOf[JList[JList[JFloat]]]
      result
    }
  }

  def predict3d(input: JList[JList[JList[JFloat]]]): JList[JList[JFloat]] = {
    timing(s"predict for input") {
      val samples = transfer3dInputToSampleArray(input)
      val result = predictor.predict(samples).toList
        .map(_.asInstanceOf[Tensor[Float]].toArray().toList)
        .asJava.asInstanceOf[JList[JList[JFloat]]]
      result
    }
  }
}

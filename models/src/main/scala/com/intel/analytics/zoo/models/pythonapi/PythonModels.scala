/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.zoo.models.pythonapi

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame}
import com.intel.analytics.zoo.models.Predictor
import org.apache.log4j.Logger

import scala.reflect.ClassTag

object PythonModels {

  def ofFloat(): PythonBigDL[Float] = new PythonModels[Float]()

  def ofDouble(): PythonBigDL[Double] = new PythonModels[Double]()

  val logger = Logger.getLogger(getClass)
}

class PythonModels[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL[T] {

   def modelZooPredict(model: AbstractModule[Activity, Activity, T],
     imageFrame: ImageFrame,
     outputLayer: String = null,
     shareBuffer: Boolean = false,
     predictKey: String = ImageFeature.predict): ImageFrame = {
     Predictor.predict(model, imageFrame, outputLayer, shareBuffer, predictKey)
   }
}
